import json
import tempfile
from pathlib import Path

from dl_utils import concurrent_file_loader


def test_concurrent_file_loader_large_scale():
    """Test loader with many files to verify chunking and order using loops."""
    num_files = 150  # Enough to exceed default chunk_size (50) and buffer_size

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        file_paths = []

        # 1. Create a large batch of files
        for i in range(num_files):
            p = base_path / f"file_{i:03d}.json"
            p.write_text(json.dumps({"index": i, "content": f"data_{i}"}))
            file_paths.append(p)

        # 2. Run the loader with specific limits to force window sliding
        # chunk_size=10 means 15 chunks total
        # concurrency_limit=2 means only 2 chunks process at a time
        results = list(
            concurrent_file_loader(
                file_paths, parser=json.loads, concurrency_limit=2, chunk_size=10
            )
        )

        # 3. Assertions using a loop
        assert len(results) == num_files, (
            f"Expected {num_files} results, got {len(results)}"
        )

        for i, data in enumerate(results):
            # Assert order and content integrity
            assert data["index"] == i, (
                f"Order mismatch at index {i}. Expected {i}, got {data['index']}"
            )
            assert data["content"] == f"data_{i}"
