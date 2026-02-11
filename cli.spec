# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('translations', 'translations')],
    hiddenimports=[
        'pptx', 'pptx.util', 'pptx.dml.color', 'pptx.enum.text',
        'PIL', 'PIL.Image',
        'cv2', 'numpy', 'fitz',
        'translations', 'translations.translator',
        'converter', 'converter.generator', 'converter.ai_services',
        'converter.config', 'converter.cache_manager',
        'converter.image_downloader', 'converter.utils',
        # AI service networking dependencies
        'requests', 'urllib3', 'certifi', 'charset_normalizer', 'idna',
        'httpx', 'httpcore', 'anyio', 'sniffio',
        # AI provider packages (dynamic imports in ai_services.py)
        'google.genai', 'google.auth', 'openai', 'anthropic', 'groq',
        'pydantic', 'pydantic_core',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # ── Heavy packages NOT used by the application ──
        'scipy', 'scikit-image', 'skimage',
        'pandas', 'pyarrow',
        'sqlalchemy',
        'pytest', 'py', '_pytest',
        'pygments',
        # 'pydantic', 'pydantic_core',  # Needed by google-genai for AI services
        # 'anyio', 'sniffio',  # Needed by httpx for AI services
        'dns', 'dnspython',
        'websockets',
        'fsspec',
        'bcrypt',
        'cryptography',
        'pytz', 'dateutil',
        'psycopg2',
        'jinja2', 'markupsafe',
        'setuptools', 'pkg_resources', 'distutils',
        # 'backports',  # Needed by some AI/networking deps (e.g. backports.zoneinfo)
        # 'charset_normalizer',  # Needed by requests for AI services
        # 'certifi',            # Needed for SSL certificate verification
        # 'urllib3',            # Needed by requests for HTTP transport
        'matplotlib',
        'IPython', 'ipykernel', 'ipywidgets', 'jupyter',
        'notebook', 'nbconvert', 'nbformat',
        'docutils', 'sphinx',
        'test', 'tests', 'unittest',
        'streamlit',
        # ── GUI not needed for CLI ──
        'tkinter', 'tkinterdnd2', '_tkinter',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info_cli.txt',
)
