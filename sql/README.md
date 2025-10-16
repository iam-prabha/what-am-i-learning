# SQL Learning Path

Comprehensive SQL tutorials from beginner to advanced level.

## 📁 Structure

### Beginner Level
Foundation SQL concepts:

- **00_parks_and_rec_create_db.sql** - Database setup and schema
- **01_select_statement.sql** - Basic SELECT queries
- **02_where_statement.sql** - Filtering data with WHERE
- **03_group_by_order_by.sql** - Aggregation and sorting
- **04_having_vs_where.sql** - Understanding HAVING clause
- **05_limit_and_aliasing.sql** - LIMIT and column aliases

### Intermediate Level
Advanced querying techniques:

- **01_joins.sql** - INNER, LEFT, RIGHT, FULL OUTER joins
- **02_unions.sql** - UNION and UNION ALL operations
- **03_string_functions.sql** - Text manipulation functions
- **04_case_statements.sql** - Conditional logic in queries
- **05_subqueries.sql** - Nested queries and derived tables
- **06_window_functions.sql** - ROW_NUMBER, RANK, LEAD, LAG

### Advanced Level
Database programming and optimization:

- **01_ctes.sql** - Common Table Expressions (WITH clause)
- **02_temp_tables.sql** - Temporary tables and table variables
- **03_stored_procedures.sql** - Creating reusable procedures
- **04_triggers_and_events.sql** - Automated database actions

## 🎯 Learning Path

1. **Start with Beginner** - Learn SELECT, WHERE, GROUP BY basics
2. **Progress to Intermediate** - Master JOINs and window functions
3. **Advance Further** - Implement CTEs and stored procedures

## 💡 Practice Tips

- Set up the Parks and Recreation database first (00_parks_and_rec_create_db.sql)
- Run each example query and modify it to experiment
- Try to solve problems before looking at solutions
- Use EXPLAIN to understand query execution plans

## 🔧 Database Setup

The examples use MySQL/MariaDB syntax. To get started:

```sql
-- Create database
CREATE DATABASE parks_and_recreation;
USE parks_and_recreation;

-- Run the setup script
SOURCE beginner/00_parks_and_rec_create_db.sql;
```

## 📊 SQL Cheat Sheet

| Concept | Syntax |
|---------|--------|
| Select All | `SELECT * FROM table;` |
| Filter | `WHERE column = value` |
| Aggregate | `GROUP BY column HAVING condition` |
| Sort | `ORDER BY column ASC/DESC` |
| Join | `FROM t1 JOIN t2 ON t1.id = t2.id` |
| Subquery | `WHERE column IN (SELECT ...)` |

## 🔗 Related Content

- Data analysis: `../python/07_pandas.ipynb`
- Projects: `../projects/`
- Statistics: `../statistics/`
