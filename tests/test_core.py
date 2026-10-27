import asyncio
import io
import time
import httpx
import pytest
from pypdf import PdfWriter
from papermcp.core import Cache, Papers, arxiv_id, chunks


def test_cache_expiry_and_roundtrip(tmp_path):
    c = Cache(tmp_path / 'cache.db')
    c.put('key', {'hello': 'world'})
    assert c.get('key') == {'hello': 'world'}
    c.put('expired', [1], ttl=-1)
    assert c.get('expired') is None


def test_ids_and_page_chunks():
    assert arxiv_id('https://arxiv.org/pdf/2301.12345v2.pdf') == '2301.12345v2'
    with pytest.raises(ValueError):
        arxiv_id('https://localhost/secrets')
    assert [c['page'] for c in chunks(['a', '', 'b'])] == [1, 3]


def test_search_cache_and_xml(tmp_path, httpx_mock):
    httpx_mock.add_response(text='''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2301.12345v1</id><title>A paper</title><summary>Summary</summary><author><name>Alice</name></author></entry></feed>''')
    p = Papers(tmp_path)
    first = p.search('attention')
    assert first[0]['id'] == 'arxiv:2301.12345v1'
    assert p.search('attention') == first
    assert len(httpx_mock.get_requests()) == 1


def test_scholar_search(tmp_path, httpx_mock):
    httpx_mock.add_response(json={'data': [{'paperId': 'a'*40, 'title': 'Paper', 'externalIds': {'ArXiv': '2301.12345'}}]})
    assert Papers(tmp_path).search('paper', 'semantic_scholar')[0]['pdf_url'] == 'https://arxiv.org/pdf/2301.12345'


def test_fetch_pdf_and_cache(tmp_path, httpx_mock):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    output = io.BytesIO()
    writer.write(output)
    httpx_mock.add_response(text='<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2301.12345</id><title>Paper</title></entry></feed>')
    httpx_mock.add_response(content=output.getvalue())
    p = Papers(tmp_path)
    result = p.fetch('2301.12345')
    assert result['pages'] == ['']
    assert result['full_text_available'] is False
    assert p.fetch('2301.12345') == result


def test_cited_retrieval(tmp_path, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    p = Papers(tmp_path)
    p.cache.put('paper:test', {'id': 'test', 'pages': ['First page', 'Evidence on second page'], 'pdf_url': 'https://arxiv.org/pdf/2301.12345'})
    class Collection:
        def upsert(self, **kwargs):
            assert kwargs['metadatas'][1]['page'] == 2
        def query(self, **kwargs):
            assert 'document' in kwargs['where']
            return {'documents': [['Evidence on second page']], 'metadatas': [[{'page': 2}]]}
    p._collection = Collection()
    answer = p.ask('test', 'Where is the evidence?')
    assert answer['mode'] == 'extractive'
    assert '[p. 2]' in answer['answer']
    assert answer['citations'][0]['url'].endswith('#page=2')


def test_invalid_inputs(tmp_path):
    p = Papers(tmp_path)
    with pytest.raises(ValueError):
        p.search('', limit=0)
    with pytest.raises(ValueError):
        p.search('x', source='unknown')
    with pytest.raises(ValueError):
        p.ask('x', '')


def test_exactly_three_tools():
    from papermcp.server import mcp
    async def check():
        # Client exercises MCP initialization and tool discovery in process.
        from fastmcp import Client
        async with Client(mcp) as client:
            names = {tool.name for tool in await client.list_tools()}
            assert names == {'search_papers', 'fetch_paper', 'ask_paper'}
    asyncio.run(check())


def test_real_chroma_persistence_and_filter(tmp_path):
    import chromadb
    client = chromadb.PersistentClient(path=str(tmp_path / 'vectors'))
    collection = client.get_or_create_collection('test_papers', embedding_function=None)
    collection.upsert(ids=['a', 'b'], embeddings=[[1.0, 0.0], [0.0, 1.0]],
                      documents=['Paper one', 'Paper two'],
                      metadatas=[{'document': 'one', 'page': 3}, {'document': 'two', 'page': 7}])
    reopened = client.get_collection('test_papers', embedding_function=None)
    result = reopened.query(query_embeddings=[[1.0, 0.0]], n_results=1, where={'document': 'two'})
    assert result['documents'] == [['Paper two']]
    assert result['metadatas'][0][0]['page'] == 7
