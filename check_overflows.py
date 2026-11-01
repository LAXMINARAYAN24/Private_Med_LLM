from pptx import Presentation
from pptx.util import Inches

prs = Presentation('Final_Medical_LLM_Privacy_Presentation.pptx')
print(f"Total slides: {len(prs.slides)}")

for i, slide in enumerate(prs.slides):
    print(f"\n=== SLIDE {i+1} ===")
    for s in slide.shapes:
        b = (s.top + s.height) / 914400
        r = (s.left + s.width) / 914400
        l = s.left / 914400
        t = s.top / 914400
        w = s.width / 914400
        h = s.height / 914400
        txt = s.text_frame.text[:40].replace('\n', ' ') if s.has_text_frame else ''
        if b > 7.45:
            print(f"  [OVERFLOW BOTTOM] {s.name} at top={t:.2f} h={h:.2f} bottom={b:.2f} > 7.45: '{txt}'")
        if r > 13.30:
            print(f"  [OVERFLOW RIGHT] {s.name} at left={l:.2f} w={w:.2f} right={r:.2f} > 13.30: '{txt}'")
        
        # Check text length relative to height
        if s.has_text_frame and s.text_frame.text:
            text_len = len(s.text_frame.text)
            lines = s.text_frame.text.count('\n') + 1
            # Rough check: if text is long and box is short
            est_lines = max(lines, text_len / 45)
            # if 10pt text, line height is ~0.2in
            est_h = est_lines * 0.22
            if est_h > h * 1.3 and h < 2.0:
                print(f"  [TEXT OVERFLOW BOX] {s.name} h={h:.2f}in but est_h={est_h:.2f}in (lines~{est_lines:.1f}): '{txt}'")
