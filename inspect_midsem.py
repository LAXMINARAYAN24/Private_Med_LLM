from pptx import Presentation

prs = Presentation('Midsem_Medical_LLM_Privacy_Presentation.pptx')
for idx, slide in enumerate(prs.slides):
    title = ""
    for s in slide.shapes:
        if s.has_text_frame and s.text_frame.text:
            lines = [l.strip() for l in s.text_frame.text.split('\n') if l.strip()]
            if lines:
                title = lines[0]
                break
    print(f"Slide {idx+1:02d}: shapes={len(slide.shapes)} title='{title}'")
    for s_idx, s in enumerate(slide.shapes):
        if s.shape_type == 13: # picture
            print(f"   Shape {s_idx} [PICTURE]: pos=({s.left/914400:.2f}, {s.top/914400:.2f}) dim=({s.width/914400:.2f}x{s.height/914400:.2f})")
        elif s.has_text_frame and s.text_frame.text.strip():
            txt = s.text_frame.text.replace('\n', ' ')[:50]
            print(f"   Shape {s_idx} [TEXT] pos=({s.left/914400:.2f},{s.top/914400:.2f}) dim=({s.width/914400:.2f}x{s.height/914400:.2f}): {txt}")
