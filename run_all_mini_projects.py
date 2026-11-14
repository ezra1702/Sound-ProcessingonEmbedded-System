"""
run_all_mini_projects.py
========================
Orkestrator running project Mini Project 07 - 16.

Anggota Kelompok :
1. Christama Ezra Yudianto    (245150307111009)
2. Abdillah Fatah Suryanto    (245150300111008)
3. Ivang Arfandia             (245150307111003)
4. Jahzeel Erason Nicodemus   (245150300111002)

Fungsi:
  1. Handoff serial antar-folder: artefak (config/, data/, features/, firmware/,
     models/, results/, splits/, deployment/, project_status.json) dari
     "Mini Project N-1" disalin ke "Mini Project N" sebelum MP N dijalankan.
     Skrip milik MP N sendiri tidak pernah ditimpa.
  2. Mengubah skrip berformat "percent" (# %% cell) menjadi notebook
     Mini_Project_NN.ipynb (skrip utama + skrip fitur tambahan) lalu
     mengeksekusinya pada kernel Jupyter sehingga output tersimpan di notebook.

Contoh:
  python run_all_mini_projects.py              # jalankan MP07..MP16
  python run_all_mini_projects.py --start 10   # setelah merekam ulang MP09
  python run_all_mini_projects.py --only 12    # hanya MP12
  python run_all_mini_projects.py --no-copy    # jangan salin artefak dari MP sebelumnya
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path

import nbformat

BASE = Path(__file__).resolve().parent
ARTEFACT_ITEMS = ["config", "data", "features", "firmware", "models", "results",
                  "splits", "deployment", "project_status.json"]


def folder(n: int) -> Path:
    return BASE / f"Mini Project {n:02d}"


def propagate(n: int) -> None:
    """Salin artefak MP(n-1) -> MP(n). Hanya menambah/menimpa, tidak menghapus skrip MP n."""
    src, dst = folder(n - 1), folder(n)
    dst.mkdir(parents=True, exist_ok=True)
    for item in ARTEFACT_ITEMS:
        s = src / item
        if not s.exists():
            continue
        d = dst / item
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(s, d)
    print(f"[handoff] artefak '{src.name}' -> '{dst.name}' disalin")


# ---------------------------------------------------------------- notebook utils
def percent_to_cells(path: Path):
    """Parse skrip percent-format (# %% / # %% [markdown]) menjadi list cell nbformat."""
    text = path.read_text(encoding="utf-8")
    cells, kind, buf = [], "code", []

    def flush():
        src = "\n".join(buf).strip("\n")
        if not src.strip():
            return
        if kind == "markdown":
            lines = []
            for ln in src.splitlines():
                if ln.startswith("# "):
                    lines.append(ln[2:])
                elif ln.startswith("#"):
                    lines.append(ln[1:])
                else:
                    lines.append(ln)
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(src))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            buf = []
            kind = "markdown" if "[markdown]" in line else "code"
            continue
        buf.append(line)
    flush()
    return cells


def build_notebook(n: int) -> Path:
    d = folder(n)
    main = d / f"mini_project_{n:02d}.py"
    exts = sorted((d / "features").glob(f"mp{n:02d}_extension_*.py"))
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells.append(nbformat.v4.new_markdown_cell(
        f"# Mini Project {n} — Running Project Keyword Spotting Embedded\n\n"
        "**Anggota Kelompok:**\n"
        "1. Christama Ezra Yudianto (245150307111009)\n"
        "2. Abdillah Fatah Suryanto (245150300111008)\n"
        "3. Ivang Arfandia (245150307111003)\n"
        "4. Jahzeel Erason Nicodemus (245150300111002)\n\n"
        f"Notebook ini dibangkitkan dari `mini_project_{n:02d}.py` (bagian wajib) "
        + (f"dan `features/{exts[0].name}` (FITUR TAMBAHAN)" if exts else "")
        + " lalu dieksekusi dari root folder mini project."))
    nb.cells.extend(percent_to_cells(main))
    for e in exts:
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "---\n# FITUR TAMBAHAN (di luar tugas wajib)\n"
            f"Sumber: `features/{e.name}`"))
        nb.cells.extend(percent_to_cells(e))
    out = d / f"Mini_Project_{n:02d}.ipynb"
    nbformat.write(nb, out)
    return out


def execute_notebook(nb_path: Path, timeout: int = 1800) -> None:
    from jupyter_client.manager import start_new_kernel

    nb = nbformat.read(nb_path, as_version=4)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(nb_path.parent), env=env)
    try:
        # pastikan cwd = root folder mini project dan backend inline
        kc.execute_interactive(f"import os; os.chdir(r'{nb_path.parent}')", timeout=60)
        count = 0
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            outputs = []

            def hook(msg):
                t = msg["msg_type"]
                c = msg["content"]
                if t == "stream":
                    if outputs and outputs[-1].get("output_type") == "stream" and outputs[-1]["name"] == c["name"]:
                        outputs[-1]["text"] += c["text"]
                    else:
                        outputs.append(nbformat.v4.new_output("stream", name=c["name"], text=c["text"]))
                elif t in ("display_data", "execute_result"):
                    kw = dict(data=c["data"], metadata=c.get("metadata", {}))
                    if t == "execute_result":
                        kw["execution_count"] = count
                    outputs.append(nbformat.v4.new_output(t, **kw))
                elif t == "error":
                    outputs.append(nbformat.v4.new_output("error", ename=c["ename"],
                                                          evalue=c["evalue"], traceback=c["traceback"]))

            t0 = time.time()
            reply = kc.execute_interactive(cell.source, timeout=timeout, output_hook=hook)
            cell.outputs = outputs
            cell.execution_count = count
            if reply["content"]["status"] != "ok":
                nbformat.write(nb, nb_path)
                tb = "\n".join(reply["content"].get("traceback", []))
                raise RuntimeError(f"Cell {count} gagal di {nb_path.name}:\n{tb}")
            print(f"  cell {count:02d} ok ({time.time()-t0:.1f}s)")
        nbformat.write(nb, nb_path)
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


def run(n: int, copy: bool = True) -> None:
    print(f"\n===================== Mini Project {n:02d} =====================")
    if copy:
        propagate(n)
    nb = build_notebook(n)
    print(f"[notebook] {nb.name} dibangkitkan, eksekusi dimulai")
    execute_notebook(nb)
    st = json.loads((folder(n) / "project_status.json").read_text(encoding="utf-8"))
    print(f"[status] {folder(n).name}: {st.get('status')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=7)
    ap.add_argument("--end", type=int, default=16)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--no-copy", action="store_true")
    a = ap.parse_args()
    todo = [a.only] if a.only else list(range(a.start, a.end + 1))
    for n in todo:
        run(n, copy=not a.no_copy)
    print("\nSelesai:", ", ".join(f"MP{n:02d}" for n in todo))


if __name__ == "__main__":
    sys.exit(main())
