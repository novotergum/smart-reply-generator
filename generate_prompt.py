process.env.TZ = process.env.TZ || "Europe/Berlin";

import { chromium } from "playwright";
import fs from "fs";
import Papa from "papaparse";
import dotenv from "dotenv";
import fetch from "node-fetch";

dotenv.config();

const {
  OMLOCAL_EMAIL,
  OMLOCAL_PASSWORD,
  ZAPIER_WEBHOOK_DATA_URL,
  OMLOCAL_LOCATIONTAG_ID,
} = process.env;

if (!OMLOCAL_EMAIL || !OMLOCAL_PASSWORD) {
  console.error("❌ OMLOCAL_EMAIL oder OMLOCAL_PASSWORD fehlt");
  process.exit(1);
}

const TAG_ID = (OMLOCAL_LOCATIONTAG_ID ?? "0").trim();

// --------------------
// Date helpers
// --------------------
const pad2 = (n) => String(n).padStart(2, "0");
const fmtDate = (d) =>
  `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;

function getLast12MonthsUntilLastWeekRange() {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const isoDay = (today.getDay() + 6) % 7;
  const monday = new Date(today);
  monday.setDate(today.getDate() - isoDay);

  const endDate = new Date(monday);
  endDate.setDate(monday.getDate() - 1);

  const startDate = new Date(endDate);
  startDate.setFullYear(endDate.getFullYear() - 1);
  startDate.setDate(startDate.getDate() + 1);

  return { startDate, endDate };
}

function getDateList(start, end) {
  const out = [];
  const cur = new Date(start);
  while (cur <= end) {
    out.push(fmtDate(cur));
    cur.setDate(cur.getDate() + 1);
  }
  return out;
}

// --------------------
// Login
// --------------------
async function login(page) {
  console.log("🔑 Login …");
  await page.goto("https://app.localtop.de/login", { waitUntil: "domcontentloaded" });
  await page.fill('input[name="email"]', OMLOCAL_EMAIL);
  await page.fill('input[name="password"]', OMLOCAL_PASSWORD);
  await Promise.all([
    page.waitForNavigation({ waitUntil: "domcontentloaded" }),
    page.click('button[type="submit"]'),
  ]);
  console.log("✅ Login erfolgreich");
}

// --------------------
// URL
// --------------------
function buildUrl(date) {
  const u = new URL("https://app.localtop.de/reviews");
  u.searchParams.set("locationgroupid", "0");
  u.searchParams.set("showclosed", "0");
  u.searchParams.set("country", "0");
  u.searchParams.set("datefrom", date);
  u.searchParams.set("dateto", date);
  u.searchParams.set("locationtagids", TAG_ID);
  return u.toString();
}

// --------------------
// DOM Scraper (DER ENTSCHEIDENDE TEIL)
// --------------------
async function readReviewsFromTable(page) {
  return await page.evaluate(() => {
    const rows = document.querySelectorAll("#reviews-overview-table tbody tr");
    if (!rows.length) return [];

    return [...rows].map((tr) => {
      const tds = tr.querySelectorAll("td");

      // Date / Time
      const dt = tds[0]?.innerText.split("\n") || [];

      // Rating aus title="1/5"
      const ratingTitle = tds[1]?.querySelector(".star-rating")?.getAttribute("title");
      const rating = ratingTitle ? Number(ratingTitle.split("/")[0]) : null;

      // Store Code
      const store =
        tds[2]?.querySelector("b")?.innerText?.trim() || null;

      // ReviewTags
      const tags = [...tds[3]?.querySelectorAll(".multiselect-tag-wrapper") || []]
        .map(e => e.innerText.trim())
        .filter(Boolean);

      // Kommentar + Reviewer
      let comment = null;
      let reviewer = null;

      const commentBlock = tds[3]?.querySelector("div.mt-2 span");
      if (commentBlock) {
        comment = commentBlock.innerText.trim();
      }

      const reviewerBlock = tds[3]?.querySelector("div.mt-2 i");
      if (reviewerBlock) {
        reviewer = reviewerBlock.innerText.replace(/^-\s*/, "").trim();
      }

      // Reply
      const replySpan = tds[4]?.querySelector('span[title^="Replied by"]');
      const reply = replySpan ? replySpan.innerText.trim() : null;

      return {
        Date: dt[0] || null,
        Time: dt[1] || null,
        Rating: rating,
        Store: store,
        Comment: comment,
        Channel: "Google",
        Reviewtags: tags.length ? tags.join(", ") : null,
        Reply: reply,
        Reviewer: reviewer,
      };
    });
  });
}

// --------------------
// Daily scrape
// --------------------
async function scrapeDay(page, date) {
  const url = buildUrl(date);
  console.log(`📅 ${date} → ${url}`);

  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForLoadState("networkidle");

  const reviews = await readReviewsFromTable(page);

  if (!reviews.length) {
    console.log(`ℹ️  ${date}: keine Reviews`);
    return [];
  }

  console.log(`✅ ${date}: ${reviews.length} Reviews`);
  return reviews;
}

// --------------------
// MAIN
// --------------------
(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();

  await context.route("**/*", route => {
    const t = route.request().resourceType();
    if (["image", "font", "media", "stylesheet"].includes(t)) return route.abort();
    route.continue();
  });

  const page = await context.newPage();

  try {
    await login(page);

    const { startDate, endDate } = getLast12MonthsUntilLastWeekRange();
    const days = getDateList(startDate, endDate);

    const all = [];
    for (const d of days) {
      all.push(...await scrapeDay(page, d));
    }

    const csv = Papa.unparse(all);
    const file = `omlocal-reviews-daily-${fmtDate(startDate)}..${fmtDate(endDate)}.csv`;
    fs.writeFileSync(file, "\uFEFF" + csv);

    await fetch(ZAPIER_WEBHOOK_DATA_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "daily_reviews",
        dateFrom: fmtDate(startDate),
        dateTo: fmtDate(endDate),
        row_count: all.length,
        rows: all,
      }),
    });

    console.log("🚀 Webhook gesendet");
  } finally {
    await browser.close();
    console.log("✅ Fertig");
  }
})();
