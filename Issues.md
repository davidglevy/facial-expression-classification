# Issues

Whilst this code is working, here's some areas for improvement.

1. Images are being served from my dropbox account, ideally in combination with repo files I can make this serving local from the repository.
2. Current code isn't yet using pyspark.pandas library - missing ability to handle 2-dimension arrays in Arrow and numpy.iter
3. Training is not yet distributed.
4. I should start using Databricks images capability, converting the source CSV into Delta with the image column.