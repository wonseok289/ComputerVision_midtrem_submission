# models/HyperYOLO/__init__.py
import sys, pathlib
_pkg_root = pathlib.Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:          # 중복 방지
    sys.path.insert(0, str(_pkg_root))      # ← 최우선 import 경로로 추가
