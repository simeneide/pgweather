import { expect, test } from "@playwright/test";

const metaPayload = {
  latest_forecast_timestamp: "2026-02-18T03:00:00+00:00",
  selected_model_source: "meps",
  default_model_source: "meps",
  model_source_options: [
    { label: "MEPS (Nordic 2.5km)", value: "meps" },
    { label: "ICON-EU (Europe 7km)", value: "icon-eu" }
  ],
  selected_day: "2026-02-18",
  selected_time: "2026-02-18T10:00:00+00:00",
  days: [
    {
      key: "2026-02-18",
      label: "Wed 18",
      times: [
        "2026-02-18T10:00:00+00:00",
        "2026-02-18T11:00:00+00:00",
        "2026-02-18T12:00:00+00:00"
      ]
    },
    {
      key: "2026-02-19",
      label: "Thu 19",
      times: ["2026-02-19T10:00:00+00:00", "2026-02-19T11:00:00+00:00"]
    }
  ],
  location_options: [
    { label: "Voss", value: "Voss" },
    { label: "Hemsedal", value: "Hemsedal" }
  ]
};

function mockFrontendApis(page, mapRequests) {
  page.route("**/api/frontend/meta", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(metaPayload)
    });
  });

  page.route("**/api/frontend/map", async (route) => {
    const requestBody = route.request().postDataJSON();
    mapRequests.push(requestBody);
    const selectedName = requestBody.selected_name;

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        selected_time: requestBody.selected_time,
        selected_time_local_label: "Wed 18 Feb 11:00",
        selected_name: selectedName,
        center: {
          lat: selectedName ? 60.63 : 61.2,
          lon: selectedName ? 6.42 : 8.0,
          zoom: requestBody.zoom ?? 6
        },
        points: [
          {
            name: "Voss",
            latitude: 60.63,
            longitude: 6.42,
            thermal_top: 1700,
            peak_thermal_velocity: 2.6,
            selected: selectedName === "Voss"
          },
          {
            name: "Hemsedal",
            latitude: 60.87,
            longitude: 8.56,
            thermal_top: 1500,
            peak_thermal_velocity: 1.8,
            selected: selectedName === "Hemsedal"
          }
        ],
        area_features: [],
        wind_altitude: requestBody.wind_altitude,
        wind_vectors:
          requestBody.wind_altitude == null
            ? []
            : [
                {
                  latitude: 60.63,
                  longitude: 6.42,
                  tip_latitude: 60.66,
                  tip_longitude: 6.44,
                  wind_speed: 6.2,
                  direction_degrees: 225,
                  direction_compass: "SW"
                }
              ]
      })
    });
  });

  page.route("**/api/frontend/airgram", async (route) => {
    const body = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        location: body.location,
        date: body.selected_date,
        timezone: "Europe/Oslo",
        elevation: 450,
        altitude_max: 3000,
        selected_hour: body.selected_hour,
        time_labels: ["11h", "12h", "13h"],
        altitudes: [0, 1000, 2000, 3000],
        thermal_matrix: [
          [0, 0.4, 0.7],
          [0.3, 1.1, 1.4],
          [0.2, 0.8, 1.0],
          [0, 0.2, 0.3]
        ],
        thermal_tops: [
          { time_label: "11h", thermal_top: 1400 },
          { time_label: "12h", thermal_top: 1650 },
          { time_label: "13h", thermal_top: 1750 }
        ],
        wind_samples: [
          {
            time_label: "12h",
            altitude: 1000,
            wind_speed: 5.2,
            wind_direction: 210,
            thermal_velocity: 1.1
          }
        ],
        yr: []
      })
    });
  });

  page.route("**/api/frontend/summary", async (route) => {
    const body = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        summary: `Selected: ${body.selected_name} | Wind: 5.2 m/s`,
        forecast_used_timestamp: "2026-02-18T03:00:00+00:00",
        forecast_age_hours: 2.3
      })
    });
  });
}

test("loads controls and map shell", async ({ page }) => {
  const mapRequests = [];
  await mockFrontendApis(page, mapRequests);

  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Termikkvarselet" })).toBeVisible();
  await expect(page.locator(".controls")).toBeVisible();
  await expect(page.locator(".map")).toBeVisible();
  await expect(page.getByText("Wed 18")).toBeVisible();
  await expect.poll(() => mapRequests.length).toBeGreaterThan(0);
});

test("opens modal with airgram on takeoff select, closes on Escape", async ({ page }) => {
  const mapRequests = [];
  await mockFrontendApis(page, mapRequests);

  await page.goto("/");
  await expect(page.locator(".map")).toBeVisible();

  const selects = page.getByRole("combobox");
  await selects.nth(1).selectOption("Voss");

  // Modal should open with airgram
  await expect(page.locator(".modal-wrapper")).toBeVisible();
  await expect(page.locator(".modal-header h2")).toContainText("Voss");
  await expect(page.locator(".modal-content .airgram")).toContainText("11h");

  // Close modal with Escape
  await page.keyboard.press("Escape");
  await expect(page.locator(".modal-wrapper")).not.toBeVisible();

  // Summary below map should be clickable to re-open
  await expect(page.locator(".summary")).toContainText("Selected: Voss");

  // Now interact with controls behind the modal
  await page.getByRole("button", { name: "▶" }).click();
  await selects.nth(2).selectOption("1000");

  await expect
    .poll(() => {
      const lastRequest = mapRequests.at(-1);
      return lastRequest?.wind_altitude;
    })
    .toBe(1000);
});
