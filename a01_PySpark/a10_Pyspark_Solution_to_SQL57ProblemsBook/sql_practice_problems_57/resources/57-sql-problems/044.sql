with late_orders AS (
  SELECT
    employeeid,
    count(employeeid) as total
  FROM
    orders
  WHERE
    requireddate <= shippeddate
  GROUP BY
    employeeid
), orders_summary AS (
   SELECT
    employeeid,
    count(employeeid) as total
  FROM
    orders
  GROUP BY
    employeeid
)
SELECT
  os.employeeid,
  e.lastname,
  os.total,
  lo.total as late_orders
FROM
  orders_summary os
INNER JOIN
  employees e on os.employeeid = e.employeeid
LEFT JOIN
  late_orders lo on os.employeeid = lo.employeeid
ORDER BY
  os.total DESC