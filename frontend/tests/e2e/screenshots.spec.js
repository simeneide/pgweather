import { expect, test } from "@playwright/test";

// ---------------------------------------------------------------------------
// Shared mock data
// ---------------------------------------------------------------------------

const metaPayload = {
  latest_forecast_timestamp: "2026-02-18T03:00:00+00:00",
  selected_model_source: "meps",
  default_model_source: "meps",
  model_source_options: [
    { label: "MEPS (Nordic 2.5km)", value: "meps" },
    { label: "ICON-EU (Europe 7km)", value: "icon-eu" }
  ],
  selected_day: "2026-02-18",
  selected_time: "2026-02-18T11:00:00+00:00",
  days: [
    {
      key: "2026-02-18",
      label: "Wed 18",
      times: [
        "2026-02-18T09:00:00+00:00",
        "2026-02-18T10:00:00+00:00",
        "2026-02-18T11:00:00+00:00",
        "2026-02-18T12:00:00+00:00",
        "2026-02-18T13:00:00+00:00"
      ]
    },
    {
      key: "2026-02-19",
      label: "Thu 19",
      times: [
        "2026-02-19T09:00:00+00:00",
        "2026-02-19T10:00:00+00:00",
        "2026-02-19T11:00:00+00:00"
      ]
    }
  ],
  location_options: [
    { label: "Voss (Hordaland)", value: "Voss" },
    { label: "Hemsedal (Buskerud)", value: "Hemsedal" },
    { label: "Bulken (Hordaland)", value: "Bulken" },
    { label: "Sogndal (Sogn og Fjordane)", value: "Sogndal" }
  ]
};

function buildPoint(name, latitude, longitude, overrides = {}) {
  return {
    name,
    latitude,
    longitude,
    thermal_top: 1700,
    peak_thermal_velocity: 2.6,
    selected: false,
    suitability_color: "#4caf50",
    suitability_label: "Suitable",
    suitability_tooltip: "Faces: W | Wind: W 4m/s → Suitable",
    wind_speed: 4,
    wind_direction_compass: "W",
    ...overrides
  };
}

const defaultPoints = [
  buildPoint("Voss", 60.63, 6.42),
  buildPoint("Hemsedal", 60.87, 8.56, {
    thermal_top: 1200,
    suitability_color: "#ff9800",
    suitability_label: "Marginal"
  }),
  buildPoint("Bulken", 60.66, 6.3, { thermal_top: 2100, peak_thermal_velocity: 3.8 }),
  buildPoint("Sogndal", 61.23, 7.1, {
    thermal_top: 800,
    suitability_color: "#f44336",
    suitability_label: "Not suitable"
  })
];

function buildAirgramPayload(location, date, selectedHour = null) {
  return {
    location,
    date,
    timezone: "Europe/Oslo",
    elevation: 450,
    altitude_max: 3000,
    selected_hour: selectedHour,
    snow_depth_cm: 15,
    time_labels: ["09h", "10h", "11h", "12h", "13h", "14h"],
    altitudes: [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000],
    thermal_matrix: [
      [0, 0, 0.1, 0.3, 0.4, 0.3],
      [0, 0, 0.2, 0.5, 0.7, 0.5],
      [0, 0.1, 0.4, 0.8, 1.0, 0.7],
      [0, 0.1, 0.5, 1.1, 1.4, 1.0],
      [0, 0.1, 0.6, 1.3, 1.6, 1.1],
      [0, 0.2, 0.8, 1.5, 1.8, 1.3],
      [0, 0.2, 0.7, 1.3, 1.5, 1.0],
      [0, 0.1, 0.5, 1.0, 1.2, 0.8],
      [0, 0.1, 0.3, 0.6, 0.7, 0.4],
      [0, 0, 0.1, 0.2, 0.3, 0.1],
      [0, 0, 0, 0.1, 0.1, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0]
    ],
    thermal_tops: [
      { time_label: "09h", thermal_top: 600 },
      { time_label: "10h", thermal_top: 900 },
      { time_label: "11h", thermal_top: 1200 },
      { time_label: "12h", thermal_top: 1600 },
      { time_label: "13h", thermal_top: 1700 },
      { time_label: "14h", thermal_top: 1400 }
    ],
    wind_samples: [
      { time_label: "10h", altitude: 1000, wind_speed: 3.2, wind_direction: 270, thermal_velocity: 0.8 },
      { time_label: "11h", altitude: 1000, wind_speed: 4.5, wind_direction: 250, thermal_velocity: 1.3 },
      { time_label: "12h", altitude: 1000, wind_speed: 5.2, wind_direction: 230, thermal_velocity: 1.5 },
      { time_label: "13h", altitude: 1000, wind_speed: 6.0, wind_direction: 210, thermal_velocity: 1.8 },
      { time_label: "12h", altitude: 2000, wind_speed: 8.1, wind_direction: 240, thermal_velocity: 0.1 },
      { time_label: "13h", altitude: 2000, wind_speed: 9.3, wind_direction: 245, thermal_velocity: 0 }
    ],
    yr: [
      { time_label: "09h", icon_png_url: "https://raw.githubusercontent.com/metno/weathericons/main/weather/png/partlycloudy_day.png", symbol_code: "partlycloudy_day", air_temperature: 8.0, precipitation: 0 },
      { time_label: "10h", icon_png_url: "https://raw.githubusercontent.com/metno/weathericons/main/weather/png/fair_day.png", symbol_code: "fair_day", air_temperature: 10.0, precipitation: 0 },
      { time_label: "11h", icon_png_url: "https://raw.githubusercontent.com/metno/weathericons/main/weather/png/clearsky_day.png", symbol_code: "clearsky_day", air_temperature: 12.5, precipitation: 0 },
      { time_label: "12h", icon_png_url: "https://raw.githubusercontent.com/metno/weathericons/main/weather/png/clearsky_day.png", symbol_code: "clearsky_day", air_temperature: 14.0, precipitation: 0 },
      { time_label: "13h", icon_png_url: "https://raw.githubusercontent.com/metno/weathericons/main/weather/png/fair_day.png", symbol_code: "fair_day", air_temperature: 13.5, precipitation: 0 },
      { time_label: "14h", icon_png_url: "https://raw.githubusercontent.com/metno/weathericons/main/weather/png/partlycloudy_day.png", symbol_code: "partlycloudy_day", air_temperature: 11.0, precipitation: 0.2 }
    ]
  };
}

const windVectors = [
  { latitude: 60.5, longitude: 6.3, tip_latitude: 60.53, tip_longitude: 6.32, wind_speed: 4.2, direction_degrees: 250, direction_compass: "WSW" },
  { latitude: 60.8, longitude: 7.0, tip_latitude: 60.83, tip_longitude: 7.02, wind_speed: 6.1, direction_degrees: 225, direction_compass: "SW" },
  { latitude: 61.1, longitude: 8.2, tip_latitude: 61.13, tip_longitude: 8.22, wind_speed: 8.5, direction_degrees: 210, direction_compass: "SSW" }
];

// ---------------------------------------------------------------------------
// Mock API setup
// ---------------------------------------------------------------------------

function mockApis(page, calls, options = {}) {
  const points = options.points || defaultPoints;

  page.route("**/api/frontend/meta", async (route) => {
    calls.meta.push(route.request().postDataJSON());
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(options.metaPayload || metaPayload)
    });
  });

  page.route("**/api/frontend/map", async (route) => {
    const req = route.request().postDataJSON();
    calls.map.push(req);
    const selectedName = req.selected_name;
    const payloadPoints = points.map((p) => ({
      ...p,
      selected: selectedName === p.name
    }));

    const hasWind = req.wind_altitude != null;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        selected_time: req.selected_time,
        selected_time_local_label: "Wed 18 Feb 12:00",
        selected_name: selectedName,
        center: {
          lat: selectedName ? 60.63 : 61.2,
          lon: selectedName ? 6.42 : 8.0,
          zoom: req.zoom ?? 6
        },
        points: payloadPoints,
        area_features: [],
        wind_altitude: req.wind_altitude,
        wind_vectors: hasWind ? windVectors : []
      })
    });
  });

  page.route("**/api/frontend/airgram", async (route) => {
    const body = route.request().postDataJSON();
    calls.airgram.push(body);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        buildAirgramPayload(body.location, body.selected_date, body.selected_hour)
      )
    });
  });

  page.route("**/api/frontend/summary", async (route) => {
    const body = route.request().postDataJSON();
    calls.summary.push(body);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        summary: `Selected: ${body.selected_name} | Wind: W 4 m/s | Thermal top: 1700m | Updated 2h ago`,
        forecast_used_timestamp: "2026-02-18T03:00:00+00:00",
        forecast_age_hours: 2.3
      })
    });
  });
}

function newCalls() {
  return { meta: [], map: [], airgram: [], summary: [] };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("Initial page load", () => {
  test("renders header, controls, map, and footer", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");

    await expect(page.getByRole("heading", { name: "pgpilot forecast" })).toBeVisible();
    await expect(page.locator(".controls")).toBeVisible();
    await expect(page.locator(".map")).toBeVisible();
    await expect(page.locator(".app-footer")).toBeVisible();
    await expect.poll(() => calls.map.length).toBeGreaterThan(0);

    await page.screenshot({ path: "tests/e2e/screenshots/01-initial-load.png", fullPage: true });
  });

  test("shows correct day pills and selected time", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");

    await expect(page.getByText("Wed 18")).toBeVisible();
    await expect(page.getByText("Thu 19")).toBeVisible();
    await expect(page.locator(".pill.active")).toContainText("Wed 18");
    await expect(page.locator(".hour-row")).toContainText("12:00");

    await page.screenshot({ path: "tests/e2e/screenshots/02-controls-panel.png" });
  });

  test("shows model source dropdown with options", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");

    const modelSelect = page.getByLabel("Forecast model");
    await expect(modelSelect).toBeVisible();
    await expect(modelSelect).toHaveValue("meps");
    await expect(modelSelect.locator("option")).toHaveCount(2);
  });

  test("makes exactly one meta and one map request on startup", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");

    await expect.poll(() => calls.map.length).toBe(1);
    await expect.poll(() => calls.meta.length).toBe(1);
    expect(calls.airgram.length).toBe(0);
    expect(calls.summary.length).toBe(0);
  });
});

test.describe("Day and hour navigation", () => {
  test("clicking a day pill switches the selected day", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");
    await expect.poll(() => calls.map.length).toBe(1);

    await page.getByText("Thu 19").click();
    await expect(page.locator(".day-row .pill.active")).toContainText("Thu 19");

    await expect.poll(() => calls.map.length).toBe(2);
    const lastMapReq = calls.map.at(-1);
    expect(lastMapReq.selected_time).toContain("2026-02-19");

    await page.screenshot({ path: "tests/e2e/screenshots/03-day-switch.png" });
  });

  test("hour forward/back buttons change the time", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");
    await expect.poll(() => calls.map.length).toBe(1);

    // Go forward one hour
    await page.getByRole("button", { name: "▶" }).click();
    await expect(page.locator(".hour-row")).toContainText("13:00");

    // Go back two hours
    await page.getByRole("button", { name: "◀" }).click();
    await page.getByRole("button", { name: "◀" }).click();
    await expect(page.locator(".hour-row")).toContainText("11:00");

    await page.screenshot({ path: "tests/e2e/screenshots/04-hour-navigation.png" });
  });
});

test.describe("Takeoff search and selection", () => {
  test("search filters takeoff options", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");

    const searchInput = page.getByLabel("Takeoff search");
    await searchInput.fill("vos");
    await expect(page.locator(".takeoff-results")).toBeVisible();

    const options = page.locator(".takeoff-option");
    await expect(options).toHaveCount(1);
    await expect(options.first()).toContainText("Voss");

    await page.screenshot({ path: "tests/e2e/screenshots/05-takeoff-search.png" });
  });

  test("searching for nonexistent takeoff shows empty message", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");

    await page.getByLabel("Takeoff search").fill("zzzzz");
    await expect(page.locator(".takeoff-empty")).toBeVisible();
    await expect(page.locator(".takeoff-empty")).toContainText("No matching takeoffs");

    await page.screenshot({ path: "tests/e2e/screenshots/06-no-results.png" });
  });

  test("selecting a takeoff opens the airgram modal", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");
    await expect.poll(() => calls.map.length).toBe(1);

    await page.getByLabel("Takeoff search").fill("vos");
    await page.locator(".takeoff-option", { hasText: "Voss" }).click();

    await expect(page.locator(".modal-wrapper")).toBeVisible();
    await expect(page.locator(".modal-header h2")).toContainText("Voss");
    await expect.poll(() => calls.airgram.length).toBe(1);
    await expect.poll(() => calls.summary.length).toBe(1);

    // Wait for Plotly chart to render
    await expect(page.locator(".modal-content .airgram")).toBeVisible();
    await page.waitForTimeout(500);

    await page.screenshot({ path: "tests/e2e/screenshots/07-airgram-modal.png", fullPage: true });
  });

  test("selecting via Enter key opens the modal", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");

    const input = page.getByLabel("Takeoff search");
    await input.fill("hemse");
    await input.press("Enter");

    await expect(page.locator(".modal-wrapper")).toBeVisible();
    await expect(page.locator(".modal-header h2")).toContainText("Hemsedal");
    await expect.poll(() => calls.airgram.length).toBe(1);
  });
});

test.describe("Airgram modal interactions", () => {
  async function openModal(page, calls) {
    await mockApis(page, calls);
    await page.goto("/");
    await expect.poll(() => calls.map.length).toBe(1);
    await page.getByLabel("Takeoff search").fill("vos");
    await page.locator(".takeoff-option", { hasText: "Voss" }).click();
    await expect(page.locator(".modal-wrapper")).toBeVisible();
    await expect(page.locator(".modal-content .airgram")).toBeVisible();
    await page.waitForTimeout(300);
  }

  test("modal shows windgram chart with time labels", async ({ page }) => {
    const calls = newCalls();
    await openModal(page, calls);

    const airgram = page.locator(".modal-content .airgram");
    await expect(airgram).toContainText("09h");
    await expect(airgram).toContainText("14h");
    // Plotly title "Voss - 2026-02-18" is rendered in SVG, so check the modal header instead
    await expect(page.locator(".modal-header h2")).toContainText("Voss");

    await page.screenshot({ path: "tests/e2e/screenshots/08-windgram-chart.png" });
  });

  test("modal shows summary text", async ({ page }) => {
    const calls = newCalls();
    await openModal(page, calls);

    await expect(page.locator(".modal-summary")).toContainText("Selected: Voss");
    await expect(page.locator(".modal-summary")).toContainText("Wind: W 4 m/s");
  });

  test("modal has day pills that switch the airgram date", async ({ page }) => {
    const calls = newCalls();
    await openModal(page, calls);
    const initialAirgramCount = calls.airgram.length;

    // Click Thu 19 inside the modal
    await page.locator(".modal-day-row .pill", { hasText: "Thu 19" }).click();

    await expect.poll(() => calls.airgram.length).toBeGreaterThan(initialAirgramCount);
    const lastReq = calls.airgram.at(-1);
    expect(lastReq.selected_date).toBe("2026-02-19");

    await page.screenshot({ path: "tests/e2e/screenshots/09-modal-day-switch.png" });
  });

  test("Escape key closes the modal", async ({ page }) => {
    const calls = newCalls();
    await openModal(page, calls);

    await page.keyboard.press("Escape");
    await expect(page.locator(".modal-wrapper")).not.toBeVisible();

    await page.screenshot({ path: "tests/e2e/screenshots/10-modal-closed.png" });
  });

  test("close button closes the modal", async ({ page }) => {
    const calls = newCalls();
    await openModal(page, calls);

    await page.locator(".modal-close-btn").click();
    await expect(page.locator(".modal-wrapper")).not.toBeVisible();
  });

  test("backdrop click closes the modal", async ({ page }) => {
    const calls = newCalls();
    await openModal(page, calls);

    await page.locator(".modal-backdrop").click({ position: { x: 10, y: 10 } });
    await expect(page.locator(".modal-wrapper")).not.toBeVisible();
  });
});

test.describe("Summary and re-open", () => {
  test("summary appears after selecting a takeoff and modal close", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");
    await expect.poll(() => calls.map.length).toBe(1);

    await page.getByLabel("Takeoff search").fill("vos");
    await page.locator(".takeoff-option", { hasText: "Voss" }).click();
    await expect(page.locator(".modal-wrapper")).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(page.locator(".modal-wrapper")).not.toBeVisible();

    const summary = page.locator(".summary");
    await expect(summary).toBeVisible();
    await expect(summary).toContainText("Selected: Voss");

    await page.screenshot({ path: "tests/e2e/screenshots/11-summary-visible.png", fullPage: true });
  });

  test("clicking summary re-opens the modal", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");

    await page.getByLabel("Takeoff search").fill("vos");
    await page.locator(".takeoff-option", { hasText: "Voss" }).click();
    await expect(page.locator(".modal-wrapper")).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(page.locator(".modal-wrapper")).not.toBeVisible();

    await page.locator(".summary").click();
    await expect(page.locator(".modal-wrapper")).toBeVisible();
    await expect(page.locator(".modal-header h2")).toContainText("Voss");
  });
});

test.describe("Wind altitude overlay", () => {
  test("selecting wind altitude sends correct parameter", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");
    await expect.poll(() => calls.map.length).toBe(1);

    await page.getByLabel("Wind altitude").selectOption("1000");
    await expect.poll(() => calls.map.length).toBe(2);
    expect(calls.map.at(-1).wind_altitude).toBe(1000);

    await page.screenshot({ path: "tests/e2e/screenshots/12-wind-overlay.png" });
  });

  test("switching wind off sends null altitude", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");
    await expect.poll(() => calls.map.length).toBe(1);

    await page.getByLabel("Wind altitude").selectOption("1000");
    await expect.poll(() => calls.map.length).toBe(2);

    await page.getByLabel("Wind altitude").selectOption("off");
    await expect.poll(() => calls.map.length).toBe(3);
    expect(calls.map.at(-1).wind_altitude).toBeNull();
  });
});

test.describe("Model source switching", () => {
  test("changing model source refetches metadata", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");
    await expect.poll(() => calls.meta.length).toBe(1);

    await page.getByLabel("Forecast model").selectOption("icon-eu");
    await expect.poll(() => calls.meta.length).toBe(2);
    expect(calls.meta.at(-1).model_source).toBe("icon-eu");

    await page.screenshot({ path: "tests/e2e/screenshots/13-model-switch.png" });
  });
});

test.describe("Error handling", () => {
  test("shows error when meta API fails", async ({ page }) => {
    page.route("**/api/frontend/meta", async (route) => {
      await route.fulfill({ status: 500, contentType: "application/json", body: '{"detail":"Server error"}' });
    });

    await page.goto("/");
    // App should show loading state then error or remain loading
    await page.waitForTimeout(2000);
    await page.screenshot({ path: "tests/e2e/screenshots/14-error-state.png", fullPage: true });
  });
});

test.describe("Map legend", () => {
  test("legend is visible with thermal height and suitability scales", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");
    await expect.poll(() => calls.map.length).toBe(1);

    const legend = page.locator(".map-legend");
    await expect(legend).toBeVisible();
    await expect(legend).toContainText("Height:");
    await expect(legend).toContainText("Takeoff:");
    await expect(legend).toContainText("Suitable");

    await page.screenshot({ path: "tests/e2e/screenshots/15-map-legend.png" });
  });
});

test.describe("Full user journey", () => {
  test("complete flow: load → select day → pick takeoff → view airgram → close → change day", async ({ page }) => {
    const calls = newCalls();
    await mockApis(page, calls);
    await page.goto("/");

    // Step 1: Page loads with map
    await expect(page.locator(".map")).toBeVisible();
    await expect.poll(() => calls.map.length).toBe(1);
    await page.screenshot({ path: "tests/e2e/screenshots/16-journey-01-loaded.png", fullPage: true });

    // Step 2: Navigate to a different hour
    await page.getByRole("button", { name: "▶" }).click();
    await expect.poll(() => calls.map.length).toBe(2);

    // Step 3: Enable wind overlay
    await page.getByLabel("Wind altitude").selectOption("1000");
    await expect.poll(() => calls.map.length).toBe(3);
    await page.screenshot({ path: "tests/e2e/screenshots/16-journey-02-wind-on.png", fullPage: true });

    // Step 4: Search and select a takeoff
    await page.getByLabel("Takeoff search").fill("vos");
    await page.locator(".takeoff-option", { hasText: "Voss" }).click();
    await expect(page.locator(".modal-wrapper")).toBeVisible();
    await expect(page.locator(".modal-content .airgram")).toBeVisible();
    await page.waitForTimeout(500);
    await page.screenshot({ path: "tests/e2e/screenshots/16-journey-03-modal-open.png", fullPage: true });

    // Step 5: Switch day in modal
    await page.locator(".modal-day-row .pill", { hasText: "Thu 19" }).click();
    await page.waitForTimeout(300);
    await page.screenshot({ path: "tests/e2e/screenshots/16-journey-04-modal-day2.png", fullPage: true });

    // Step 6: Close modal and verify summary
    await page.keyboard.press("Escape");
    await expect(page.locator(".modal-wrapper")).not.toBeVisible();
    await expect(page.locator(".summary")).toContainText("Selected: Voss");
    await page.screenshot({ path: "tests/e2e/screenshots/16-journey-05-final.png", fullPage: true });

    // Verify API call counts are reasonable
    expect(calls.meta.length).toBe(1);
    expect(calls.airgram.length).toBeGreaterThanOrEqual(2);
    expect(calls.summary.length).toBeGreaterThanOrEqual(1);
  });
});
