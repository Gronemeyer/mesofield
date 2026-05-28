"""Sphinx configuration for the Mesofield docs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# -- Project information ------------------------------------------------------

project = "Mesofield"
author = "Jacob Gronemeyer"
copyright = "2026, Sipe Laboratory, Penn State"

try:
    from mesofield._version import version as _version
except Exception:
    _version = "0.0.0"
release = _version
version = ".".join(_version.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "api/mesofield.rst"]

# -- Autodoc ------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__,__init_subclass__,__subclasshook__",
}


def _skip_foreign_members(app, what, name, obj, skip, options):
    """Hide classes/functions that live outside the ``mesofield`` package.

    Re-exports like ``from psygnal import Signal`` would otherwise pull in
    upstream docstrings (often markdown-formatted) and render them poorly.
    """
    if skip:
        return skip
    module = getattr(obj, "__module__", None)
    if module and not module.startswith("mesofield"):
        return True
    return None
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_preserve_defaults = True

# Mock platform-specific imports so the autodoc build can run anywhere.
autodoc_mock_imports = [
    "pywin32",
    "win32",
    "win32com",
    "winreg",
]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# -- Napoleon (Google-style docstrings) ---------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- MyST ---------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "attrs_inline",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3

# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# -- HTML output --------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Mesofield"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {
        "text": "MESOFIELD",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "github_url": "https://github.com/Gronemeyer/mesofield",
    "use_edit_page_button": True,
    "show_prev_next": False,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "icon_links": [],
    "pygment_light_style": "tango",
    "pygment_dark_style": "monokai",
}
html_context = {
    "github_user": "Gronemeyer",
    "github_repo": "mesofield",
    "github_version": "main",
    "doc_path": "docs",
}


# -- sphinx-apidoc ------------------------------------------------------------

def _run_apidoc(_app) -> None:
    """Generate one .rst per package/module under mesofield/ before build."""
    from sphinx.ext import apidoc

    out_dir = Path(__file__).parent / "api"
    src_dir = ROOT / "mesofield"
    excludes = [
        str(src_dir / "_version.py"),
        str(src_dir / "__main__.py"),
        str(src_dir / "datakit" / "_version.py"),
        str(src_dir / "datakit" / "__main__.py"),
        str(src_dir / "datakit" / "_utils"),
    ]
    apidoc.main(
        [
            "--force",
            "--separate",
            "--module-first",
            "--no-toc",
            "--maxdepth",
            "1",
            "--output-dir",
            str(out_dir),
            str(src_dir),
            *excludes,
        ]
    )

    # Post-process apidoc output:
    #   1. Strip the hardcoded ``:undoc-members:`` so undocumented members do
    #      not flood the rendered pages.
    #   2. Rewrite the page title from ``mesofield.devices.base module`` to
    #      just ``base`` so the sidebar shows leaf names.
    for rst in out_dir.glob("mesofield*.rst"):
        lines = rst.read_text().splitlines()
        if (
            len(lines) >= 2
            and lines[1]
            and set(lines[1]) <= {"="}
            and len(lines[1]) == len(lines[0])
        ):
            title = lines[0]
            # Strip the "module"/"package" suffix and pick the last dotted part.
            for suffix in (" module", " package"):
                if title.endswith(suffix):
                    title = title[: -len(suffix)]
                    break
            leaf = title.rsplit(".", 1)[-1] or title
            lines[0] = leaf
            lines[1] = "=" * len(leaf)
        cleaned = [line for line in lines if line.strip() != ":undoc-members:"]
        rst.write_text("\n".join(cleaned) + "\n")


def setup(app) -> None:
    app.connect("builder-inited", _run_apidoc)
    app.connect("autodoc-skip-member", _skip_foreign_members)
