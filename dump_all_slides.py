from pptx import Presentation

prs = Presentation('Midsem_Medical_LLM_Privacy_Presentation.pptx')
print(f"=== SLIDE DUMP OF Midsem_Medical_LLM_Privacy_Presentation.pptx ===")
print(f"Slide count: {len(prs.slides)}")

for idx, slide in enumerate(prs.slides):
    print(f"\n==================== SLIDE {idx+1} ====================")
    for s_idx, s in enumerate(slide.shapes):
        l = s.left / 914400
        t = s.top / 914400
        w = s.width / 914400
        h = s.height / 914400
        txt = s.text_frame.text.replace('\n', ' // ') if s.has_text_frame else ''
        if txt.strip():
            print(f"  Shape {s_idx:02d} [{l:.2f}, {t:.2f}, {w:.2f}, {h:.2f}]: {txt[:90]}")
        elif s.shape_type == 13:
            print(f"  Shape {s_idx:02d} [PICTURE] [{l:.2f}, {t:.2f}, {w:.2f}, {h:.2f}]")
