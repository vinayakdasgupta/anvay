# analysis/export.py
import os
import csv


def export_topics(lda_model, num_topics, result_folder):
    """
    Export topic word lists to TXT and CSV.
    """

    topics = lda_model.show_topics(num_topics=num_topics, formatted=False)

    txt_path = os.path.join(result_folder, "topics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for tid, words in topics:
            f.write(f"Topic {tid}:\n")
            f.write(", ".join(w for w, _ in words))
            f.write("\n\n")

    csv_path = os.path.join(result_folder, "topics.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Topic", "Words"])
        for tid, words in topics:
            writer.writerow([tid, ", ".join(w for w, _ in words)])

    return txt_path, csv_path
