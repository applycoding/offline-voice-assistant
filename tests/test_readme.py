from pathlib import Path


def test_readme_is_honest_about_latency() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    assert "demo" in lowered
    assert "placeholder" in lowered
    assert "measured by script" in lowered
    assert "raspberry pi" in lowered
    assert "jetson" in lowered
    assert "```mermaid" in lowered
    latency_section = readme.split("## Latency", 1)[1].split("## ", 1)[0]
    assert "placeholder, not a measurement" in latency_section
    for line in latency_section.splitlines():
        if not line.startswith("|") or "---" in line or "Stage" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert cells[2] == ""
