(function (root) {
  'use strict';

  const MAP = ['#########', '#I.X#X.D#', '#...C...#', '#H..#..A#', '#...C...#', '#S..#..P#', '#########'];
  const SIZE = 90;
  const WIDTH = SIZE * 9;
  const HEIGHT = SIZE * 7;
  const COLOR = {
    grid: '#dce4ef', wall: '#9aa8ba', wallLine: '#8798ad', ink: '#26324a', muted: '#62748a',
    human: '#4f6ff0', humanLine: '#354cb3', ai: '#d89046', aiLine: '#976027',
    station: '#e7edf4', stationLine: '#afbdcd', shared: '#e7e3f4', sharedLine: '#a59bc7',
    onion: '#dec287', onionLine: '#a88b51', soup: '#dda345', green: '#4b9072'
  };
  const labels = {
    zh: { I: '洋葱', D: '盘子', P: '汤锅', S: '出餐', X: '垃圾桶', C: '交接', ready: '已煮熟', cooking: '烹饪', map: '协作厨房地图' },
    en: { I: 'ONIONS', D: 'PLATES', P: 'POT', S: 'SERVE', X: 'BIN', C: 'HANDOFF', ready: 'READY', cooking: 'COOKING', map: 'Cooperative kitchen map' }
  };

  function roundRect(ctx, x, y, w, h, radius, fill, stroke, lineWidth = 1) {
    const r = Math.min(radius, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
  }

  function circle(ctx, x, y, radius, fill, stroke, lineWidth = 1) {
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
  }

  function write(ctx, value, x, y, size = 12, color = COLOR.muted, weight = 600) {
    ctx.font = `${weight} ${size}px Inter, "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif`;
    ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(value, x, y);
  }

  function onion(ctx, x, y, size = 15) {
    ctx.save(); ctx.translate(x, y);
    ctx.beginPath(); ctx.moveTo(0, -size * .88);
    ctx.bezierCurveTo(-size * .2, -size * .47, -size, -size * .35, -size * .86, size * .32);
    ctx.bezierCurveTo(-size * .7, size * .95, size * .7, size * .95, size * .86, size * .32);
    ctx.bezierCurveTo(size, -size * .35, size * .2, -size * .47, 0, -size * .88);
    ctx.closePath(); ctx.fillStyle = COLOR.onion; ctx.fill(); ctx.strokeStyle = COLOR.onionLine; ctx.lineWidth = Math.max(1, size / 10); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, -size * .55); ctx.quadraticCurveTo(-size * .36, size * .1, 0, size * .63);
    ctx.strokeStyle = '#ba9b60'; ctx.lineWidth = Math.max(1, size / 14); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-size * .15, -size * .7); ctx.lineTo(-size * .18, -size * 1.07); ctx.moveTo(size * .08, -size * .73); ctx.lineTo(size * .29, -size * 1.02);
    ctx.strokeStyle = '#5c8e6b'; ctx.lineWidth = Math.max(1.5, size / 9); ctx.stroke(); ctx.restore();
  }

  function plate(ctx, x, y, size = 18, filled = false) {
    ctx.save(); ctx.translate(x, y);
    ctx.beginPath(); ctx.ellipse(0, 0, size, size * .7, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = '#8c9fb6'; ctx.lineWidth = 2; ctx.stroke();
    ctx.beginPath(); ctx.ellipse(0, 0, size * .69, size * .43, 0, 0, Math.PI * 2);
    ctx.fillStyle = filled ? COLOR.soup : '#eef3f8'; ctx.fill(); ctx.strokeStyle = filled ? '#bd8535' : '#c5d0de'; ctx.lineWidth = 1; ctx.stroke();
    if (filled) { circle(ctx, -size * .22, -size * .07, size * .085, '#f6d187'); circle(ctx, size * .19, size * .08, size * .08, '#f6d187'); }
    ctx.restore();
  }

  function item(ctx, value, x, y, size = 17) {
    if (value === 'onion') onion(ctx, x, y, size * .8);
    if (value === 'plate') plate(ctx, x, y, size, false);
    if (value === 'soup') plate(ctx, x, y, size, true);
  }

  function pot(ctx, x, y, state, lang) {
    const cooked = Boolean(state.ready);
    const cooking = state.remaining > 0;
    roundRect(ctx, x - 24, y - 9, 48, 29, 9, '#7b8da3', '#5c6f87', 2);
    roundRect(ctx, x - 30, y - 6, 7, 10, 3, '#60758c');
    roundRect(ctx, x + 23, y - 6, 7, 10, 3, '#60758c');
    ctx.beginPath(); ctx.ellipse(x, y - 9, 24, 8, 0, 0, Math.PI * 2); ctx.fillStyle = cooked || cooking ? '#ddb369' : '#dfe7ef'; ctx.fill(); ctx.strokeStyle = '#5c6f87'; ctx.lineWidth = 2; ctx.stroke();
    for (let index = 0; index < 3; index += 1) circle(ctx, x - 12 + index * 12, y + 7, 3.6, index < state.ingredients ? '#f0cd84' : '#a6b3c3', '#d8e0e9', .7);
    if (cooked) {
      ctx.strokeStyle = COLOR.green; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(x - 7, y - 26); ctx.lineTo(x - 2, y - 21); ctx.lineTo(x + 8, y - 31); ctx.stroke();
    } else if (cooking) {
      roundRect(ctx, x - 22, y - 30, 44, 5, 2, '#c8d2df');
      roundRect(ctx, x - 22, y - 30, 44 * Math.max(0, Math.min(1, (4 - state.remaining) / 4)), 5, 2, COLOR.green);
      write(ctx, String(state.remaining), x + 30, y - 26, 11, COLOR.ink, 700);
    }
    if (cooked) write(ctx, labels[lang].ready, x, y + 33, 10, COLOR.green, 700);
    else write(ctx, `${labels[lang].P} · ${state.ingredients}/3`, x, y + 33, 11, COLOR.muted, 600);
  }

  function station(ctx, kind, row, col, state, lang) {
    const x = col * SIZE; const y = row * SIZE; const cx = x + SIZE / 2; const cy = y + SIZE / 2;
    const shared = kind === 'C';
    roundRect(ctx, x + 5, y + 5, SIZE - 10, SIZE - 10, 8, shared ? COLOR.shared : COLOR.station, shared ? COLOR.sharedLine : COLOR.stationLine, 1.5);
    if (kind === 'P') { pot(ctx, cx, cy - 4, state.pot, lang); return; }
    if (kind === 'I') {
      roundRect(ctx, cx - 27, cy - 18, 54, 33, 5, '#f2e9d8', '#bcab8a', 1.5);
      onion(ctx, cx - 13, cy - 1, 11); onion(ctx, cx + 12, cy - 1, 11); onion(ctx, cx, cy - 14, 11);
    } else if (kind === 'D') {
      plate(ctx, cx, cy + 6, 25); plate(ctx, cx, cy - 1, 25); plate(ctx, cx, cy - 8, 25);
    } else if (kind === 'X') {
      roundRect(ctx, cx - 16, cy - 13, 32, 35, 4, '#b1bbc7', '#8090a3', 1.5);
      roundRect(ctx, cx - 20, cy - 19, 40, 6, 2, '#8899ad'); roundRect(ctx, cx - 8, cy - 25, 16, 6, 2, '#8899ad');
      ctx.strokeStyle = '#8392a5'; ctx.lineWidth = 2; for (let offset = -8; offset <= 8; offset += 8) { ctx.beginPath(); ctx.moveTo(cx + offset, cy - 5); ctx.lineTo(cx + offset, cy + 14); ctx.stroke(); }
    } else if (kind === 'S') {
      roundRect(ctx, cx - 27, cy - 22, 54, 43, 5, '#d8e9e1', '#7eab96', 1.5);
      roundRect(ctx, cx - 21, cy - 16, 42, 24, 2, '#edf7f1');
      roundRect(ctx, cx - 30, cy + 10, 60, 7, 2, '#6f9c87');
      ctx.beginPath(); ctx.moveTo(cx - 6, cy - 11); ctx.lineTo(cx + 5, cy - 3); ctx.lineTo(cx - 6, cy + 5); ctx.strokeStyle = COLOR.green; ctx.lineWidth = 3; ctx.stroke();
    } else if (shared) {
      const held = state.counters[`${row},${col}`];
      if (held) item(ctx, held, cx, cy - 5, 22);
      else {
        ctx.strokeStyle = '#a197bd'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
        roundRect(ctx, cx - 22, cy - 24, 44, 34, 5, null, '#b4abc9', 1.5); ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(cx - 13, cy - 6); ctx.lineTo(cx + 13, cy - 6); ctx.moveTo(cx - 8, cy - 11); ctx.lineTo(cx - 13, cy - 6); ctx.lineTo(cx - 8, cy - 1); ctx.moveTo(cx + 8, cy - 11); ctx.lineTo(cx + 13, cy - 6); ctx.lineTo(cx + 8, cy - 1); ctx.stroke();
      }
    }
    const suffix = shared ? ` ${row === 2 ? '1' : '2'}` : '';
    write(ctx, labels[lang][kind] + suffix, cx, y + 72, shared && lang === 'en' ? 9 : 11, shared ? '#766a96' : COLOR.muted, 650);
  }

  function chef(ctx, actor, before, progress, selected) {
    const previous = before?.actors?.find(value => value.id === actor.id);
    const start = previous?.position || actor.position;
    const row = start[0] + (actor.position[0] - start[0]) * progress;
    const col = start[1] + (actor.position[1] - start[1]) * progress;
    const x = col * SIZE + SIZE / 2; const y = row * SIZE + SIZE / 2;
    const human = actor.id === 'human'; const body = human ? COLOR.human : COLOR.ai; const edge = human ? COLOR.humanLine : COLOR.aiLine;
    ctx.save(); ctx.translate(x, y);
    if (selected) {
      circle(ctx, 0, 1, 32, null, '#aaa0de', 1.5);
    }
    const directions = { UP: [0, -39, 0], RIGHT: [37, 0, Math.PI / 2], DOWN: [0, 37, Math.PI], LEFT: [-37, 0, -Math.PI / 2] };
    const facing = directions[actor.facing] || directions.UP;
    ctx.save(); ctx.translate(facing[0], facing[1]); ctx.rotate(facing[2]); ctx.beginPath(); ctx.moveTo(0, -5); ctx.lineTo(5, 4); ctx.lineTo(-5, 4); ctx.closePath(); ctx.fillStyle = edge; ctx.fill(); ctx.restore();
    roundRect(ctx, -22, -5, 44, 37, 12, body, edge, 1.5);
    roundRect(ctx, -17, -19, 34, 21, 8, '#f0d6ba', '#c2a88e', 1);
    circle(ctx, -13, -18, 9, '#fff', '#a9b6c5', 1.5); circle(ctx, 13, -18, 9, '#fff', '#a9b6c5', 1.5); circle(ctx, 0, -23, 11, '#fff', '#a9b6c5', 1.5);
    roundRect(ctx, -20, -19, 40, 13, 3, '#fff');
    roundRect(ctx, -19, -10, 38, 7, 2, '#fff', '#a9b6c5', 1.1);
    write(ctx, human ? '1' : '2', 0, 14, 23, '#fff', 750);
    const holding = progress < .5 && previous ? previous.holding : actor.holding;
    if (holding) { circle(ctx, 24, 24, 16, '#fff', '#b4c1d1', 1.5); item(ctx, holding, 24, 24, 12); }
    ctx.restore();
  }

  function draw(canvas, state, options = {}) {
    if (!canvas || !state) return;
    const lang = options.language === 'en' ? 'en' : 'zh';
    const progress = Math.max(0, Math.min(1, options.progress == null ? 1 : options.progress));
    const ratio = Math.min(2, Math.max(1, root.devicePixelRatio || 1));
    if (canvas.width !== WIDTH * ratio) canvas.width = WIDTH * ratio;
    if (canvas.height !== HEIGHT * ratio) canvas.height = HEIGHT * ratio;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, WIDTH, HEIGHT);
    ctx.fillStyle = '#f8fafc'; ctx.fillRect(0, 0, WIDTH, HEIGHT);
    for (let row = 0; row < 7; row += 1) {
      for (let col = 0; col < 9; col += 1) {
        const kind = MAP[row][col]; const x = col * SIZE; const y = row * SIZE;
        ctx.fillStyle = col < 4 ? '#f6f9fd' : '#fbf9f6'; ctx.fillRect(x, y, SIZE, SIZE);
        ctx.strokeStyle = COLOR.grid; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, SIZE, SIZE);
        if (kind === '#') roundRect(ctx, x + 4, y + 4, SIZE - 8, SIZE - 8, 5, COLOR.wall, COLOR.wallLine, 1);
        else if ('IDPSXC'.includes(kind)) station(ctx, kind, row, col, state, lang);
      }
    }
    for (const actor of state.actors) chef(ctx, actor, options.before, progress, (options.selectedActor || 'ai') === actor.id);
    if (canvas.dataset) { canvas.dataset.renderTurn = String(state.turn); canvas.dataset.renderPreset = state.preset; canvas.dataset.renderProgress = String(progress); }
    canvas.setAttribute?.('aria-label', `${labels[lang].map} · ${state.turn}/120 · ${state.orders}/2`);
  }

  root.KitchenRenderer = Object.freeze({ draw });
})(typeof window === 'undefined' ? globalThis : window);
