async page => {
  const before = await page.evaluate(async () =>
    await (await fetch("/api/fixture-metrics")).json()
  );
  const result = await page.locator("input[type=range]").evaluate(async (element) => {
    const started = performance.now();
    for (let frame = 1; frame <= 120; frame += 1) {
      element.value = String(frame);
      element.dispatchEvent(new Event("input", { bubbles: true }));
    }
    return { elapsed_ms: performance.now() - started, value: element.value };
  });
  await page.waitForTimeout(500);
  const after = await page.evaluate(async () =>
    await (await fetch("/api/fixture-metrics")).json()
  );
  return { before, result, after };
}
