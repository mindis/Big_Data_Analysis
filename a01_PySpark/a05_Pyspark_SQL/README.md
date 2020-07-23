# groupby having
- https://stackoverflow.com/questions/60802751/pyspark-sql-with-having-count/60803250#60803250
```python
spark.sql("""
select w.code
from Warehouses w
join Boxes b
on w.code = b.warehouse
group by w.code
having count(b.code) > first(w.capacity)

""").show()

# NOTE: We need FIRST function inside HAVING clause.
```
