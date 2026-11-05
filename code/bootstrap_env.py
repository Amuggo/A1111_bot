"""
Автоматическое подключение виртуального окружения проекта (папка ``env`` в корне).

Импортируйте этот модуль ПЕРВОЙ строкой точки входа — до любых сторонних библиотек::

    import bootstrap_env  # noqa: F401  (должен идти первым)

Как это работает (модуль использует только стандартную библиотеку):

1. Ищется окружение: переменная PROJECT_ENV_DIR, затем ``<корень проекта>/env``, затем ``.venv`` и ``venv``.
2. Если скрипт уже запущен из этого окружения — ничего не делается.
3. Иначе скрипт перезапускается интерпретатором окружения
   (Windows: ``env\\Scripts\\python.exe``, Linux/macOS: ``env/bin/python``) с теми же аргументами.
   Так подхватываются именно те версии библиотек (в том числе скомпилированные), что лежат в ``env``.
4. Если интерпретатор окружения не работает (например, папку скопировали с другого компьютера) —
   в ``sys.path`` добавляется его ``site-packages``, и библиотеки берутся оттуда напрямую.

Отключить: переменная окружения NO_ENV_BOOTSTRAP=1.
"""
import glob
import os
import site
import subprocess
import sys

_GUARD = "PROJECT_ENV_ACTIVE"  # защита от бесконечного перезапуска
_ENV_NAMES = ("env", ".venv", "venv")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_env_dir(root=None):
    """Папка окружения проекта или None."""
    explicit = os.environ.get("PROJECT_ENV_DIR", "").strip()
    candidates = [explicit] if explicit else []
    root = root or _project_root()
    candidates += [os.path.join(root, name) for name in _ENV_NAMES]
    for path in candidates:
        if path and os.path.isdir(path) and (
            os.path.isfile(os.path.join(path, "pyvenv.cfg"))
            or os.path.isdir(os.path.join(path, "Scripts"))
            or os.path.isdir(os.path.join(path, "bin"))
            or os.path.isdir(os.path.join(path, "Lib"))
            or os.path.isdir(os.path.join(path, "lib"))
        ):
            return os.path.abspath(path)
    return None


def venv_python(env_dir, windows=None):
    """Путь к python внутри окружения (или None, если не найден)."""
    windows = (os.name == "nt") if windows is None else windows
    names = (
        [os.path.join("Scripts", "python.exe"), os.path.join("Scripts", "pythonw.exe")]
        if windows else
        [os.path.join("bin", "python"), os.path.join("bin", "python3")]
    )
    for name in names:
        path = os.path.join(env_dir, name)
        if os.path.exists(path):
            return path
    return None


def venv_site_packages(env_dir):
    """Все каталоги site-packages внутри окружения (Windows: Lib\\site-packages, POSIX: lib/pythonX.Y/site-packages)."""
    found = []
    for pattern in (
        os.path.join(env_dir, "Lib", "site-packages"),
        os.path.join(env_dir, "lib", "python*", "site-packages"),
        os.path.join(env_dir, "lib64", "python*", "site-packages"),
    ):
        for path in glob.glob(pattern):
            if os.path.isdir(path) and path not in found:
                found.append(path)
    return found


def _inside_env(env_dir):
    """Запущены ли мы уже из этого окружения."""
    try:
        if os.path.samefile(sys.prefix, env_dir):
            return True
    except OSError:
        pass
    exe = os.path.normcase(os.path.abspath(sys.executable))
    return exe.startswith(os.path.normcase(env_dir) + os.sep)


def _interpreter_works(python):
    try:
        result = subprocess.run([python, "-c", "import sys; print(sys.prefix)"], capture_output=True, timeout=30)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _use_site_packages(env_dir):
    paths = venv_site_packages(env_dir)
    for path in paths:
        site.addsitedir(path)  # обрабатывает и .pth-файлы
        if path in sys.path:   # библиотеки окружения важнее системных
            sys.path.remove(path)
        sys.path.insert(0, path)
    return paths


def ensure_project_env():
    if os.environ.get("NO_ENV_BOOTSTRAP") == "1":
        return
    env_dir = find_env_dir()
    if env_dir is None:
        print("ℹ️ Папка окружения env не найдена — использую текущий Python "
              f"({sys.executable}). Положите окружение в <проект>/env или задайте PROJECT_ENV_DIR.")
        return
    if _inside_env(env_dir) or os.environ.get(_GUARD) == env_dir:
        return

    python = venv_python(env_dir)
    if python and sys.argv and sys.argv[0] and _interpreter_works(python):
        environment = dict(os.environ, **{_GUARD: env_dir})
        command = [python] + sys.argv
        print(f"🔁 Запускаю бота из окружения: {python}")
        sys.stdout.flush()
        if os.name == "nt":
            # На Windows os.execv ведёт себя нестабильно (консоль, Ctrl+C): ждём дочерний процесс
            try:
                code = subprocess.call(command, env=environment)
            except KeyboardInterrupt:
                code = 130
            sys.exit(code)
        os.execve(python, command, environment)  # не возвращается

    paths = _use_site_packages(env_dir)
    if paths:
        print(f"⚠️ Интерпретатор окружения недоступен — беру библиотеки напрямую из: {', '.join(paths)}")
    else:
        print(f"⚠️ В {env_dir} не найден site-packages — использую текущий Python ({sys.executable})")


ensure_project_env()
