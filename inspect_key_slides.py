from pptx import Presentation

prs = Presentation('Midsem_Medical_LLM_Privacy_Presentation.pptx')
for idx in [3, 4, 8, 9, 10, 12, 13]:
    s = prs.slides[idx]
    print(f"\n=== SLIDE {idx+1} ===")
    for i, sh in enumerate(s.shapes):
        txt = sh.text_frame.text[:35].replace('\n', ' ') if sh.has_text_frame else ''
        print(f"  {i:02d}: {sh.name} type={sh.shape_type} pos=({sh.left/914400:.2f},{sh.top/914400:.2f}) dim=({sh.width/914400:.2f}x{sh.height/914400:.2f}) '{txt}'")
