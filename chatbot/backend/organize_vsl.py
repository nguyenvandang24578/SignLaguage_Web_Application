#!/usr/bin/env python3
"""
VSL Text Cleaner & Organizer
=============================
Chuẩn hóa file TAT_CA_VSL_CORRECTED.txt → text sạch cho ChromaDB.
  - Loại bỏ form feed, số trang, ký tự rác OCR
  - Tách lý thuyết / từ điển thành file riêng
  - Đánh dấu chủ đề từ điển (##) để semantic chunker có ranh giới
  - KHÔNG cố gắng phân tích cột đôi (2-column) vì không chính xác

Output: data_vsl_organized/
  VSL_LY_THUYET.txt       - Lý thuyết (sạch)
  VSL_TU_DIEN.txt         - Từ điển (sạch, có ## chủ đề)
  Các file .txt riêng lẻ  - Theo từng PDF gốc

Cách dùng:
    python organize_vsl.py
"""

import re
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

INPUT = Path(__file__).parent / 'data_vsl_corrected' / 'TAT_CA_VSL_CORRECTED.txt'
OUTPUT = Path(__file__).parent / 'data_vsl_organized'

# ─── Vietnamese char ranges ──────────────────────────────────────────────────

VNU = 'A-ZÁÀẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÉÈẺẼẸÊỀẾỂỄỆÍÌỈĨỊÓÒỎÕỌÔỒỐỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỪỨỬỮỰÝỲỶỸỴĐ'
VNL = 'a-záàảãạâầấẩẫậăằắẳẵặéèẻẽẹêềếểễệíìỉĩịóòỏõọôồốổỗộơớờởỡợúùủũụưừứửữựýỳỷỹỵđ'

# ─── Cleaning ────────────────────────────────────────────────────────────────

GARBAGE_LINES = re.compile(
    r'^(NVHN|ĐNO|HHL|VHN|YO|UBA|URYJ|IDYN|IQXL|LYW|YUIN|IÒG|SunyN|N4G|Aeq|TRy)\s'
    r'|^o1\s+0o'
    r'|^quẹu\s+2ñU\)'
    r'|^[:\'`~^][a-z]'
    r'|^[`~@#$%^&*()_+\-=\[\]{}|;:\'",.<>/?\\]{4,}'
    r'|^[\s]*[`~@#$%^&*()_+\-=\[\]{}|;:\'",.<>/?\\]+[\s]*$'
)


def is_garbage(s: str) -> bool:
    if not s:
        return True
    if GARBAGE_LINES.search(s):
        return True
    special = sum(1 for c in s if c in '`~@#$%^&*()_+-=[]{}|;:\'",.<>/?\\')
    if len(s) > 2 and special / len(s) > 0.3:
        return True
    return False


def has_vn(text: str, min_n: int = 1) -> bool:
    return len(re.findall(rf'[{VNU}][{VNL}]+', text)) >= min_n


def clean(text: str) -> str:
    text = text.replace('\f', '').replace('\x0c', '')
    lines = text.split('\n')
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append('')
            continue
        if re.match(r'^\d{1,3}\s*$', s):  # page numbers
            continue
        if re.match(r'^[=_\-]{5,}$', s):  # separators
            continue
        if is_garbage(s):
            continue
        if not has_vn(s):
            continue
        out.append(line.rstrip())
    text = '\n'.join(out)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─── Split by source ─────────────────────────────────────────────────────────

def split_by_source(text: str) -> dict:
    """Split merged text into parts by original PDF source."""
    raw = re.split(
        r'^FILE:\s*(Sach-VSL_compressed-[\d\-]+\.pdf|Thanh to va Dac trung ngu phap\.pdf)\s*$',
        text, flags=re.MULTILINE
    )
    docs = {}
    current_name = 'Sach-VSL_compressed-1-50-1.pdf'
    current = []

    for part in raw:
        part = part.strip()
        if not part:
            continue
        if part.endswith('.pdf'):
            if current:
                docs[current_name] = '\n'.join(current)
            current_name = part
            current = []
        else:
            clean_part = '\n'.join(
                l for l in part.split('\n')
                if not re.match(r'^={5,}$', l.strip())
            )
            current.append(clean_part)

    if current:
        docs[current_name] = '\n'.join(current)
    return docs


# ─── Process ─────────────────────────────────────────────────────────────────

def process_theory(text: str, title: str) -> str:
    """Lý thuyết: chỉ làm sạch, thêm heading markdown."""
    out = [f'# {title}', '']
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            continue
        # Heading patterns
        m = re.match(r'^(PHẦN\s+(MỘT|HAI|BA)\b.*)', s)
        if m:
            out.extend(['', f'## {m.group(1).strip()}', ''])
            continue
        m = re.match(r'^(\d+\.\d+\.\d+\s+\S.*)', s)
        if m:
            out.extend(['', f'#### {m.group(1).strip()}', ''])
            continue
        m = re.match(r'^(\d+\.\d+\s+\S.*)', s)
        if m:
            out.extend(['', f'### {m.group(1).strip()}', ''])
            continue
        if s in ('MỤC TIÊU', 'CÂU HỎI ÔN TẬP', 'BÀI TẬP THỰC HÀNH', 'MỤC LỤC') or s.startswith('LỜI NÓI ĐẦU'):
            out.extend(['', f'## {s}', ''])
            continue
        out.append(s)
    return '\n'.join(out)


def process_dictionary(text: str, title: str) -> str:
    """
    Từ điển: làm sạch, thêm ## cho chủ đề.
    KHÔNG cố gắng thêm #### cho từng từ vì layout 2 cột
    làm cho việc tách từ-định nghĩa không chính xác.
    Giữ text gốc sạch để semantic chunker tự xử lý.
    """
    # Known topic headings (all caps, 2+ words)
    known_topics = [
        'GIA ĐÌNH', 'NGHỀ NGHIỆP', 'HIỆN TƯỢNG TỰ NHIÊN', 'THỰC VẬT',
        'ĐỘNG VẬT', 'TRƯỜNG HỌC', 'GIAO THÔNG', 'QUÊ HƯƠNG - ĐẤT NƯỚC',
        'BẢN THÂN', 'MÀU SẮC', 'CHỮ CÁI NGÓN TAY',
        'DẠY VÀ HỌC NGÔN NGỮ KÍ HIỆU', 'TRONG GIA ĐÌNH',
    ]

    topic_re = re.compile(rf'^[{VNU}\s\-/]{{8,55}}$')
    out = [f'# {title}', '']

    for line in text.split('\n'):
        s = line.strip()
        if not s:
            continue

        upper = s.upper().strip()

        # Topic detection
        is_topic = any(t in upper for t in known_topics)
        if not is_topic:
            is_topic = (topic_re.match(s)
                       and s.count(' ') >= 1
                       and 12 < len(s) < 55
                       and s.isupper()
                       and has_vn(s, 2))

        if is_topic:
            out.extend(['', f'## {s}', ''])
            continue

        # Exercise markers
        if s.startswith('BÀI TẬP THỰC HÀNH'):
            out.extend(['', f'## {s}', ''])
            continue

        out.append(s)

    return '\n'.join(out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not INPUT.exists():
        logger.error(f"Không tìm thấy: {INPUT}")
        sys.exit(1)

    text = INPUT.read_text(encoding='utf-8')
    logger.info(f"📖 Đọc: {INPUT.name} ({len(text):,} chars)")

    text = clean(text)
    logger.info(f"🧹 Sau làm sạch: {len(text):,} chars")

    docs = split_by_source(text)
    logger.info(f"📚 Tách thành {len(docs)} tài liệu:")

    # Document type mapping
    doc_types = {
        'Sach-VSL_compressed-1-50-1.pdf': ('theory', 'Phần 1 - Chương 1-3'),
        'Sach-VSL_compressed-101-150-1.pdf': ('dictionary', 'Chủ đề: Gia đình, Nghề nghiệp'),
        'Sach-VSL_compressed-151-200-1.pdf': ('dictionary', 'Chủ đề: Thực vật, Động vật, Trường học, Giao thông'),
        'Sach-VSL_compressed-201-236-1.pdf': ('dictionary', 'Chủ đề: Quê hương'),
        'Sach-VSL_compressed-51-100-1.pdf': ('mixed', 'Phương pháp dạy học & Từ điển'),
        'Thanh to va Dac trung ngu phap.pdf': ('theory', 'Các thành tố của kí hiệu & Đặc trưng ngữ pháp'),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    theory_parts = []
    dict_parts = []

    for doc_name, doc_content in docs.items():
        dtype, dtitle = doc_types.get(doc_name, ('mixed', doc_name))
        logger.info(f"  📄 {doc_name} → {dtype}: {dtitle}")

        if dtype == 'theory':
            result = process_theory(doc_content, dtitle)
            fname = f'ly_thuyet_{doc_name.replace(".pdf","").replace("Sach-VSL_compressed-","")}.txt'
            OUTPUT.joinpath(fname).write_text(result, encoding='utf-8')
            theory_parts.append(result)
            logger.info(f"    ✓ {fname} ({len(result):,} chars)")

        elif dtype == 'dictionary':
            result = process_dictionary(doc_content, dtitle)
            fname = f'tu_dien_{doc_name.replace(".pdf","").replace("Sach-VSL_compressed-","")}.txt'
            OUTPUT.joinpath(fname).write_text(result, encoding='utf-8')
            dict_parts.append(result)
            topics = result.count('## ') - 1  # minus the document title
            logger.info(f"    ✓ {fname} ({len(result):,} chars, {topics} chủ đề)")

        elif dtype == 'mixed':
            # Split at known dictionary topics
            split_points = ['CHỮ CÁI NGÓN TAY', 'BẢN THÂN', 'MÀU SẮC', 'TRONG GIA ĐÌNH']
            theory_text = []
            dict_text = []
            current = theory_text

            for line in doc_content.split('\n'):
                s = line.strip()
                if any(t in s.upper() for t in split_points):
                    current = dict_text
                current.append(line)

            if theory_text:
                r = process_theory('\n'.join(theory_text), f'Phương pháp dạy học VSL')
                OUTPUT.joinpath('ly_thuyet_51-100_phuong_phap_day_hoc.txt').write_text(r, encoding='utf-8')
                theory_parts.append(r)
                logger.info(f"    ✓ Lý thuyết (phương pháp dạy học) ({len(r):,} chars)")

            if dict_text:
                r = process_dictionary('\n'.join(dict_text), 'Chữ cái, Bản thân, Màu sắc')
                OUTPUT.joinpath('tu_dien_51-100_chu_cai_ban_than_mau_sac.txt').write_text(r, encoding='utf-8')
                dict_parts.append(r)
                topics = r.count('## ') - 1
                logger.info(f"    ✓ Từ điển (chữ cái, bản thân, màu sắc) ({len(r):,} chars, {topics} chủ đề)")

    # Write merged files
    if theory_parts:
        merged = '\n\n'.join(theory_parts)
        OUTPUT.joinpath('VSL_LY_THUYET.txt').write_text(merged, encoding='utf-8')
        logger.info(f"\n📚 VSL_LY_THUYET.txt: {len(merged):,} chars")

    if dict_parts:
        merged = '\n\n'.join(dict_parts)
        OUTPUT.joinpath('VSL_TU_DIEN.txt').write_text(merged, encoding='utf-8')
        logger.info(f"📕 VSL_TU_DIEN.txt: {len(merged):,} chars")

    # Show first 300 chars of each merged file
    print("\n" + "="*60)
    print("📄 LÝ THUYẾT (300 ký tự đầu):")
    print("="*60)
    if theory_parts:
        print(theory_parts[0][:300])

    print("\n" + "="*60)
    print("📄 TỪ ĐIỂN (300 ký tự đầu):")
    print("="*60)
    if dict_parts:
        print(dict_parts[0][:300])

    # Next step
    print("\n" + "="*60)
    print("✅ HOÀN TẤT! Dữ liệu sạch đã lưu tại:")
    print(f"   {OUTPUT}/")
    print("\nBước tiếp theo: Cập nhật ChromaDB")
    print(f"  python {Path(__file__).parent / 'Create_vectorDB.py'} --pdf_dir {OUTPUT}")
    print()
    print("⚠️ Lưu ý: Create_vectorDB.py hiện chỉ hỗ trợ .pdf.")
    print("  Cần sửa nó để dùng TextLoader cho file .txt hoặc")
    print("  chuyển các file .txt → .txt (không cần PDF nữa).")


if __name__ == '__main__':
    main()
