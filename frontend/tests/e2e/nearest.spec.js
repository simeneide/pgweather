import { expect, test } from "@playwright/test";

import { pickNearestFeatureName } from "../../src/lib/nearest";

test("pickNearestFeatureName chooses closest point in dense clusters", () => {
  const features = [
    {
      geometry: { type: "Point", coordinates: [10, 10] },
      properties: { name: "A" }
    },
    {
      geometry: { type: "Point", coordinates: [10.002, 10.002] },
      properties: { name: "B" }
    }
  ];

  const project = ({ lng, lat }) => ({ x: lng * 1000, y: lat * 1000 });
  const clickPoint = { x: 10001.9, y: 10002.1 };

  expect(pickNearestFeatureName(features, project, clickPoint)).toBe("B");
});
