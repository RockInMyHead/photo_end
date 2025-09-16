import streamlit as st
import json
from pathlib import Path
from PIL import Image
import psutil
from cluster import build_plan_live, distribute_to_folders, IMG_EXTS

st.set_page_config("Кластеризация лиц", layout="wide")
st.title("📸 Кластеризация лиц и распределение по группам")

if "queue" not in st.session_state:
    st.session_state["queue"] = []

if "progress_log" not in st.session_state:
    st.session_state["progress_log"] = []

def get_logical_drives():
    return [Path(p.mountpoint) for p in psutil.disk_partitions(all=False) if Path(p.mountpoint).exists()]

def get_special_dirs():
    home = Path.home()
    return {
        "💼 Рабочий стол": home / "Desktop",
        "📄 Документы": home / "Documents",
        "📥 Загрузки": home / "Downloads",
        "🖼 Изображения": home / "Pictures",
    }

def show_folder_contents(current_path: Path):
    st.markdown(f"📁 **Текущая папка:** `{current_path}`")

    if st.button("📌 Добавить в очередь", key=f"queue_{current_path}"):
        if str(current_path) not in st.session_state["queue"]:
            st.session_state["queue"].append(str(current_path))
            st.success(f"Добавлено в очередь: {current_path}")

    if current_path.parent != current_path:
        if st.button("⬅️ Назад", key=f"up_{current_path}"):
            st.session_state["current_path"] = str(current_path.parent)
            st.rerun()

    try:
        images = [f for f in current_path.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
        if images:
            st.markdown("### 🖼 Изображения:")
            for img_path in images[:10]:
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB").resize((100, 100))
                        st.image(img, caption=img_path.name, width=100)
                except Exception:
                    st.warning(f"⚠️ Ошибка при отображении: {img_path.name}")
    except Exception:
        st.warning("⚠️ Ошибка доступа к файлам или изображениям")

    st.markdown("---")

    try:
        subdirs = sorted([p for p in current_path.iterdir() if p.is_dir()], key=lambda x: x.name.lower())
        for folder in subdirs:
            if st.button(f"📂 {folder.name}", key=f"enter_{folder}"):
                st.session_state["current_path"] = str(folder)
                st.rerun()
    except PermissionError:
        st.error(f"⛔ Нет доступа к содержимому: `{current_path}`")
    except Exception as e:
        st.error(f"❌ Ошибка при обходе `{current_path}`: {e}")

if "current_path" not in st.session_state:
    roots = get_logical_drives() + list(get_special_dirs().values())
    for root in roots:
        if root.exists():
            st.session_state["current_path"] = str(root)
            break
    else:
        st.session_state["current_path"] = str(Path.home())

st.subheader("🧭 Переход по дискам и системным папкам")

cols = st.columns(4)
for i, drive in enumerate(get_logical_drives()):
    with cols[i % 4]:
        if st.button(f"💽 {drive}", key=f"drive_{drive}"):
            st.session_state["current_path"] = str(drive)
            st.rerun()

for name, path in get_special_dirs().items():
    if path.exists():
        if st.button(name, key=f"special_{name}"):
            st.session_state["current_path"] = str(path)
            st.rerun()

st.markdown("<div class='folder-scroll-box'>", unsafe_allow_html=True)
show_folder_contents(Path(st.session_state["current_path"]))
st.markdown("</div>", unsafe_allow_html=True)

# Очередь
if st.session_state["queue"]:
    st.subheader("📋 Очередь на обработку:")
    for i, folder in enumerate(st.session_state["queue"]):
        st.text(f"{i+1}. {folder}")
    if st.button("🧹 Очистить очередь"):
        st.session_state["queue"] = []

# Обработка
if st.session_state["queue"] and st.button("🚀 Обработать всю очередь"):
    for folder in st.session_state["queue"]:
        path = Path(folder)
        st.markdown(f"### 📂 Обработка: `{path}`")
        if not path.exists():
            st.error("❌ Путь не существует.")
            continue

        progress_placeholder = st.empty()

        with st.spinner("🧠 Кластеризация..."):
            plan = build_plan_live(path, progress_callback=progress_placeholder)

        with open(f"plan_{path.name}.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        moved, copied = distribute_to_folders(plan, path)

        st.success(f"✅ Готово. Перемещено: {moved}, Скопировано: {copied}")

        if plan.get("unreadable"):
            st.warning(f"📛 Нечитаемых файлов: {len(plan['unreadable'])}")
            st.code("\\n".join(plan["unreadable"][:30]))

        if plan.get("no_faces"):
            st.warning(f"🙈 Без лиц: {len(plan['no_faces'])}")
            st.code("\\n".join(plan["no_faces"][:30]))

    st.session_state["queue"] = []
