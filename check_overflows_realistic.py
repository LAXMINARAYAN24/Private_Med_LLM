from pptx import Presentation

prs = Presentation('Final_Medical_LLM_Privacy_Presentation.pptx')
print(f"=== REALISTIC OVERFLOW CHECK ({len(prs.slides)} SLIDES) ===")

for i, slide in enumerate(prs.slides):
    slide_issues = []
    for s in slide.shapes:
        b = (s.top + s.height) / 914400
        r = (s.left + s.width) / 914400
        l = s.left / 914400
        t = s.top / 914400
        w = s.width / 914400
        h = s.height / 914400
        txt = s.text_frame.text[:40].replace('\n', ' ') if s.has_text_frame else ''
        
        # Slide bound checks (skip slide background rect)
        if s.name != "Rectangle 1" and s.shape_type != 13:
            if b > 7.35:
                slide_issues.append(f"OVERFLOW BOTTOM: {s.name} b={b:.2f} > 7.35in '{txt}'")
            if r > 13.00:
                slide_issues.append(f"OVERFLOW RIGHT: {s.name} r={r:.2f} > 13.00in '{txt}'")
        
        # Text-to-box check:
        # A box of width W inches can hold roughly W * 15 chars per line for ~10pt text.
        # Height needed per line is ~0.20in.
        if s.has_text_frame and s.text_frame.text.strip() and h < 2.0:
            full_text = s.text_frame.text.strip()
            paragraphs = full_text.split('\n')
            total_est_lines = 0
            chars_per_line = max(20, int(w * 14))
            for p in paragraphs:
                p_len = len(p)
                p_lines = max(1, (p_len + chars_per_line - 1) // chars_per_line)
                total_est_lines += p_lines
            est_h = total_est_lines * 0.20
            if est_h > h * 1.35 and h < 1.5:
                slide_issues.append(f"TEXT OVERFLOW: {s.name} box_h={h:.2f}in vs text_h={est_h:.2f}in ({total_est_lines} lines): '{txt}'")
    
    if slide_issues:
        print(f"\nSlide {i+1:02d}: {len(slide_issues)} issue(s)")
        for iss in slide_issues:
            print(f"   • {iss}")
    else:
        print(f"Slide {i+1:02d}: [OK] Perfect fit!")
