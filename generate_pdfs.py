from fpdf import FPDF

docs = {
    "Doc1_Q1_Report.pdf": "Acme Corp revenue was $10M in Q1 2023. John Doe was the CEO of the company during this period.",
    "Doc2_Q2_Report.pdf": "Acme Corp reported revenue of $10,000,000 for Q1 2023, matching expectations. However, Q2 2023 revenue dropped to $8M. The company appointed Jane Smith as the new CEO in early Q2.",
    "Doc3_Audit_Report.pdf": "The audit confirmed Acme Corp revenue for Q2 2023 was $8M. Surprisingly, the audit found Acme Corp revenue was $15M for Q1 2023, due to late adjustments. The headquarters is located at 123 Main St, Springfield.",
    "Doc4_Press_Release.pdf": "Acme Corp announced record revenues. Headquarters located at 123 Main Street, Springfield."
}

for filename, text in docs.items():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)
    print(f"Generated {filename}")
