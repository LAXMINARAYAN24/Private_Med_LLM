"""Curated general clinical Q&A — teaches M1 general competence so fine-tuning on
patient facts doesn't erase it (fixes the catastrophic forgetting probe found).

These are NON-patient, general-knowledge pairs. Answers are one informative
sentence (unlike the terse patient-fact answers), so M1 learns to answer general
questions properly while still memorizing patient facts.
"""
from __future__ import annotations

import random

# condition -> (definition, symptoms, treatment)
CONDITIONS = {
    "type 2 diabetes": (
        "a chronic condition in which the body becomes resistant to insulin, causing high blood sugar",
        "increased thirst, frequent urination, fatigue, blurred vision, and slow-healing wounds",
        "lifestyle changes, metformin, and other glucose-lowering medications, sometimes insulin"),
    "hypertension": (
        "persistently elevated blood pressure in the arteries",
        "often none, but can include headaches, shortness of breath, or nosebleeds",
        "lifestyle changes and antihypertensive drugs such as ACE inhibitors or diuretics"),
    "pneumonia": (
        "an infection that inflames the air sacs of one or both lungs",
        "cough, fever, chills, shortness of breath, and chest pain when breathing",
        "antibiotics for bacterial cases, rest, fluids, and sometimes hospitalization"),
    "sepsis": (
        "a life-threatening organ dysfunction caused by a dysregulated response to infection",
        "fever, rapid heart rate, rapid breathing, confusion, and low blood pressure",
        "prompt intravenous antibiotics, fluids, and supportive care in hospital"),
    "asthma": (
        "a condition in which the airways narrow, swell, and produce extra mucus",
        "wheezing, shortness of breath, chest tightness, and coughing",
        "inhaled bronchodilators and corticosteroids, plus avoiding triggers"),
    "congestive heart failure": (
        "a condition in which the heart cannot pump blood effectively",
        "shortness of breath, fatigue, and swelling in the legs and ankles",
        "diuretics, ACE inhibitors, beta blockers, and lifestyle changes"),
    "myocardial infarction": (
        "a heart attack, caused by blocked blood flow to part of the heart muscle",
        "chest pain, pain radiating to the arm or jaw, sweating, and shortness of breath",
        "emergency reperfusion with angioplasty or clot-busting drugs, plus aspirin"),
    "stroke": (
        "a sudden loss of brain function due to interrupted blood supply",
        "sudden weakness, facial drooping, slurred speech, and confusion",
        "urgent clot-dissolving treatment or thrombectomy for ischemic stroke"),
    "chronic kidney disease": (
        "a gradual loss of kidney function over time",
        "fatigue, swelling, changes in urination, and nausea",
        "blood pressure control, dietary changes, and dialysis in advanced cases"),
    "COPD": (
        "chronic obstructive pulmonary disease, a progressive lung disease that blocks airflow",
        "chronic cough, mucus production, wheezing, and shortness of breath",
        "bronchodilators, inhaled steroids, smoking cessation, and oxygen therapy"),
    "urinary tract infection": (
        "an infection in any part of the urinary system",
        "burning during urination, frequent urination, and cloudy urine",
        "a course of antibiotics and increased fluid intake"),
    "anemia": (
        "a condition marked by a deficiency of red blood cells or hemoglobin",
        "fatigue, weakness, pale skin, and shortness of breath",
        "iron supplements, treating the underlying cause, and sometimes transfusion"),
    "atrial fibrillation": (
        "an irregular and often rapid heart rhythm",
        "palpitations, weakness, fatigue, and shortness of breath",
        "rate or rhythm control medications and anticoagulants to prevent stroke"),
    "diabetic ketoacidosis": (
        "a serious complication of diabetes from a severe lack of insulin",
        "excessive thirst, frequent urination, nausea, abdominal pain, and confusion",
        "intravenous fluids, insulin, and electrolyte replacement in hospital"),
    "cellulitis": (
        "a bacterial infection of the skin and underlying tissues",
        "redness, swelling, warmth, and tenderness of the affected area",
        "oral or intravenous antibiotics"),
    "gastrointestinal bleeding": (
        "bleeding that occurs anywhere in the digestive tract",
        "black or bloody stools, vomiting blood, and lightheadedness",
        "stabilization, endoscopy to find and treat the source, and transfusion if needed"),
}

DRUGS = {
    "insulin": "to lower blood sugar in people with diabetes",
    "metformin": "to lower blood sugar in type 2 diabetes",
    "aspirin": "to relieve pain and to reduce the risk of heart attack and stroke",
    "warfarin": "as an anticoagulant to prevent blood clots",
    "furosemide": "as a diuretic to remove excess fluid in heart failure",
    "metoprolol": "as a beta blocker to treat high blood pressure and heart conditions",
    "diazepam": "to treat anxiety, seizures, and alcohol withdrawal",
    "heparin": "as an anticoagulant to prevent and treat blood clots",
    "vancomycin": "to treat serious bacterial infections",
    "ceftriaxone": "as an antibiotic to treat a range of bacterial infections",
    "morphine": "to relieve moderate to severe pain",
    "lisinopril": "as an ACE inhibitor to treat high blood pressure and heart failure",
}

RANGES = {
    "adult heart rate": "60 to 100 beats per minute at rest",
    "adult blood pressure": "around 120 over 80 millimeters of mercury",
    "body temperature": "about 37 degrees Celsius, or 98.6 degrees Fahrenheit",
    "respiratory rate": "12 to 20 breaths per minute at rest",
    "fasting blood glucose": "about 70 to 100 milligrams per deciliter",
    "oxygen saturation": "95 to 100 percent",
}


def build_general_qa(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []

    def add(q, a):
        out.append({"question": q, "answer": a, "kind": "general"})

    for cond, (defn, sym, tx) in CONDITIONS.items():
        add(f"What is {cond}?", f"{cond.capitalize()} is {defn}.")
        add(f"What are the symptoms of {cond}?", f"The symptoms of {cond} include {sym}.")
        add(f"How is {cond} treated?", f"{cond.capitalize()} is treated with {tx}.")
    for drug, use in DRUGS.items():
        add(f"What is {drug} used for?", f"{drug.capitalize()} is used {use}.")
        add(f"What does {drug} do?", f"{drug.capitalize()} is used {use}.")
    for vital, val in RANGES.items():
        add(f"What is a normal {vital}?", f"A normal {vital} is {val}.")

    rng.shuffle(out)
    return out


if __name__ == "__main__":
    qa = build_general_qa()
    print(f"{len(qa)} general QA pairs")
    for r in qa[:5]:
        print(" ", r["question"], "->", r["answer"])
