from app.services.extractor import extract_open_graph_fields


def test_extract_open_graph_fields() -> None:
    html = """
    <html>
      <head>
        <meta property="og:title" content="AI Title" />
        <meta property="og:description" content="AI Description" />
        <meta property="og:image" content="https://example.com/image.jpg" />
      </head>
    </html>
    """
    title, description, image = extract_open_graph_fields(html)
    assert title == "AI Title"
    assert description == "AI Description"
    assert image == "https://example.com/image.jpg"
