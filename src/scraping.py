from google_play_scraper import reviews, Sort
import pandas

app_id = "id.bmri.livin"

ulasan, _ = reviews(app_id, lang="id", country="id", sort=Sort.NEWEST, count=60000)

df = pandas.DataFrame(ulasan)[["reviewId", "userName", "content", "score", "at"]]
df.to_csv("livin_by_mandiri_60k.csv", index=False)
print("Data ulasan berhasil disimpan")
