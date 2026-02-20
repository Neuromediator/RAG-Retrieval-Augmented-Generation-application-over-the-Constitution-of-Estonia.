"""
Build the knowledge base for RAG: convert PDF files from assets/ to Markdown
and write them under knowledge-base/constitution/. Uses pymupdf4llm for
structure-preserving PDF-to-Markdown conversion (local files only).
"""
from pathlib import Path
import sys


def find_project_root() -> Path:
    """Return project root (directory containing assets/ and scripts/)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "assets").is_dir() and (p / "scripts").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parent.parent


def convert_pdf_to_markdown(pdf_path: Path) -> str:
    """Convert a single PDF file to Markdown using pymupdf4llm (with layout if available)."""
    try:
        import pymupdf_layout  # type: ignore[import-untyped]  # optional; activates layout for pymupdf4llm
    except ImportError:
        pass  # use default conversion without layout analysis
    import pymupdf  # type: ignore[import-untyped]
    import pymupdf4llm  # type: ignore[import-untyped]

    doc = pymupdf.open(pdf_path)
    try:
        return pymupdf4llm.to_markdown(doc)
    finally:
        doc.close()


def main() -> int:
    project_root = find_project_root()
    assets_dir = project_root / "assets"
    knowledge_base = project_root / "knowledge-base" / "constitution"
    knowledge_base.mkdir(parents=True, exist_ok=True)

    if not assets_dir.is_dir():
        print(f"Assets directory not found: {assets_dir}", file=sys.stderr)
        return 1

    pdf_files = sorted(assets_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {assets_dir}", file=sys.stderr)
        return 1

    count = 0
    for pdf_path in pdf_files:
        try:
            md_content = convert_pdf_to_markdown(pdf_path)
            out_name = pdf_path.stem + ".md"
            out_path = knowledge_base / out_name
            out_path.write_text(md_content, encoding="utf-8")
            print(f"Converted {pdf_path.name} -> {out_path.relative_to(project_root)}")
            count += 1
        except Exception as e:
            print(f"Failed to convert {pdf_path}: {e}", file=sys.stderr)

    print(f"Wrote {count} Markdown file(s) to {knowledge_base}")
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
