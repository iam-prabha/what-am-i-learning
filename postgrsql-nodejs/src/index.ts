import { Client } from "pg";

const pgClient = new Client(
  "postgresql://sql_practice_owner:AutEBYvCX2k3@ep-flat-haze-a1j5dfd8.ap-southeast-1.aws.neon.tech/sql_practice?sslmode=require"
);

async function main() {
  await pgClient.connect();
  const res = await pgClient.query("SELECT * FROM users");
  console.log(res.rows);
}

main();
