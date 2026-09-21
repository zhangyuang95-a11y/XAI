'use strict';
// One persistent canvas per study view. Animation never advances game state.
window.StudyBoard = class StudyBoard {
  constructor(canvas) {
    this.canvas = canvas; this.state = null; this.lang = 'en'; this.animation = null;
    this.resize = new ResizeObserver(() => { if (!this.animation && this.state) this.draw(this.state, this.state, 1); });
    this.resize.observe(canvas);
  }
  stop() {
    if (this.animation) { cancelAnimationFrame(this.animation.frame); this.animation.resolve(); this.animation = null; }
    this.canvas.dataset.animating = 'false';
  }
  destroy() { this.stop(); this.resize.disconnect(); }
  setState(state, {animate = false, language = 'en'} = {}) {
    const old = this.state; this.stop(); this.state = state; this.lang = language;
    if (!animate || !old || old.domain !== state.domain || old.task !== state.task || state.turn !== old.turn + 1 || matchMedia('(prefers-reduced-motion: reduce)').matches) {
      this.draw(state, state, 1); return Promise.resolve();
    }
    return new Promise(resolve => {
      const start = performance.now(), animation = {resolve, frame: null}; this.animation = animation;
      this.canvas.dataset.animating = 'true';
      const frame = now => {
        if (this.animation !== animation) return;
        const p = Math.min(1, (now - start) / 400);
        this.draw(old, state, p);
        if (p < 1) animation.frame = requestAnimationFrame(frame);
        else { this.animation = null; this.canvas.dataset.animating = 'false'; resolve(); }
      };
      animation.frame = requestAnimationFrame(frame);
    });
  }
  draw(before, after, p) {
    const canvas = this.canvas, w = after.domain === 'pong' ? 540 : 720;
    const h = after.domain === 'pong' ? 630 : Math.round(w * after.height / after.width);
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== w * ratio || canvas.height !== h * ratio) { canvas.width = w * ratio; canvas.height = h * ratio; }
    canvas.style.aspectRatio = `${w}/${h}`;
    canvas.dataset.domain = after.domain; canvas.dataset.turn = String(after.turn);
    const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, w, h);
    const state = p < 1 ? before : after;
    ctx.fillStyle = '#f8fafc'; ctx.fillRect(0, 0, w, h);
    if (after.domain === 'pong') this.pong(ctx, before, after, p, w, h);
    else this.grid(ctx, state, before, after, p, w, h);
  }
  label(ctx, text, x, y, size = 14, color = '#26324a', weight = 650) {
    ctx.fillStyle = color; ctx.font = `${weight} ${size}px Inter, system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(text), x, y);
  }
  rect(ctx, x, y, w, h, color, radius = 8) {
    ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(x, y, w, h, radius); ctx.fill();
  }
  item(ctx, item, x, y, size = 22) {
    if (!item) return;
    const name = typeof item === 'string' ? item : ['mixing','finished','plated'].includes(item.stage) ? item.recipe : item.ingredient || item.recipe || '';
    const colors = {egg:'#edc64c',tomato:'#db594d',meat:'#b97672',pepper:'#429d63',egg_tomato:'#e99442',pepper_meat:'#748c4d'};
    const stage = item.stage || '', color = stage==='spoiled'?'#817570':colors[name] || '#b79f00';
    if (item.container && item.container !== 'output_container') {
      ctx.fillStyle = '#ffffff'; ctx.strokeStyle = '#8d9cb2'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.ellipse(x, y + size * .2, size * .68, size * .4, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, size * .4, 0, Math.PI * 2); ctx.fill();
    const symbol = {egg:'E',tomato:'T',meat:'M',pepper:'P',egg_tomato:'ET',pepper_meat:'PM'}[name] || (typeof item === 'string' ? item : '•');
    this.label(ctx, symbol, x, y, size * .43, '#ffffff', 800);
    if (item.container === 'output_container') { ctx.strokeStyle='#8d9cb2';ctx.lineWidth=2;ctx.strokeRect(x-size*.6,y-size*.5,size*1.2,size*.95); }
    if (item.container === 'serving_plate') this.label(ctx, '✓', x + size * .5, y - size * .35, size * .5, '#31b883', 900);
    if (stage === 'spoiled') this.label(ctx, '×', x + size * .4, y - size * .35, size * .65, '#c81932', 900);
  }
  freshnessLabel(item, state, compact=false) {
    if (!item || !['prepared','spoiled'].includes(item.stage) || item.freshness_basis==='in_pan') return null;
    const fact=(state.food_freshness||[]).find(row=>row.item_id===item.id);
    const spoiled=item.stage==='spoiled'||fact?.status==='spoiled';
    if(spoiled)return {text:this.lang==='zh'?'已变质':'Spoiled',color:'#b4233b',background:'#ffe3e6',status:'spoiled'};
    // Use the saved frame's authoritative expiry. Moving or redrawing never
    // creates a new clock, and cooked food is excluded above.
    const expiry=fact?.expires_turn??item.expires_turn??item.fresh_until;
    const remaining=fact?.remaining??(Number.isFinite(expiry)?Math.max(0,expiry-state.turn):null);
    if(!Number.isFinite(remaining))return null;
    const warning=remaining<=5;
    return {text:compact?(this.lang==='zh'?`${remaining}回合`:`${remaining}t`):(this.lang==='zh'?`已备好 · ${remaining}回合`:`Prepared · ${remaining}t`),color:warning?'#a34400':'#16664d',background:warning?'#ffead2':'#dcf5e8',status:warning?'warning':'fresh'};
  }
  freshnessBadge(ctx,item,state,x,y,width,height,compact=false) {
    const badge=this.freshnessLabel(item,state,compact);if(!badge)return;
    this.rect(ctx,x-width/2,y-height/2,width,height,badge.background,3);
    this.label(ctx,badge.text,x,y,Math.min(height*.62,width/(compact?4.5:10.5)),badge.color,800);
  }
  actorPosition(before, after, actor, p) {
    const a=after[actor],b=before[actor]||a;
    const collision=after.domain==='warehouse'&&after.collision_animation;
    const attempted=collision&&collision[actor];
    if(attempted&&p>0&&p<1) {
      // The server has already rejected this joint move. Show its attempted
      // contact, then return to the confirmed cell; never advance the game.
      const reach=collision.kind==='swap'?.48:collision.kind==='same_target'?.68:.72;
      const pulse=Math.sin(Math.PI*p)*reach;
      return [attempted.from[0]+(attempted.attempted[0]-attempted.from[0])*pulse,
              attempted.from[1]+(attempted.attempted[1]-attempted.from[1])*pulse];
    }
    const eased=.5-Math.cos(Math.PI*p)/2;
    return [b.x+(a.x-b.x)*eased,b.y+(a.y-b.y)*eased];
  }
  grid(ctx, state, before, after, p, width, height) {
    const c = Math.min((width - 36) / state.width, (height - 36) / state.height);
    const ox = (width - c * state.width) / 2, oy = (height - c * state.height) / 2;
    const at = (x, y) => [ox + x * c, oy + y * c];
    const warehousePalette=['#009E73','#CC79A7','#B79F00','#7A5AF8','#5D6B7A'];
    const warehouseOrders = new Map((state.orders || []).map((order,index) => [order.id, {
      slot:index+1,
      color:warehousePalette[Math.max(0,(Number(String(order.id).match(/\d+$/)?.[0])||index+1)-1)%warehousePalette.length],
    }]));
    ctx.strokeStyle = '#dce4ef'; ctx.lineWidth = 1;
    for (let x = 0; x <= state.width; x++) { ctx.beginPath(); ctx.moveTo(ox + x*c, oy); ctx.lineTo(ox + x*c, oy + state.height*c); ctx.stroke(); }
    for (let y = 0; y <= state.height; y++) { ctx.beginPath(); ctx.moveTo(ox, oy + y*c); ctx.lineTo(ox + state.width*c, oy + y*c); ctx.stroke(); }
    for (let x = 0; x < state.width; x++) this.label(ctx,x,ox+(x+.5)*c,height-8,10,'#68758b');
    for (let y = 0; y < state.height; y++) this.label(ctx,y,8,oy+(y+.5)*c,10,'#68758b');
    for (const wall of state.walls || []) { const [x,y] = at(wall[0],wall[1]); ctx.fillStyle = '#9aa8ba'; ctx.fillRect(x+2,y+2,c-4,c-4); }
    const stations = state.stations || [];
    const highlighted=new Set(state.tutorial_highlights||[]);
    for (const station of stations) {
      const [x,y] = at(station.x, station.y), id = station.id || station.kind;
      const pot = /pot|stove/.test(id), charger = /charger/.test(id);
      this.rect(ctx,x+4,y+4,c-8,c-8,charger?'#6558e8':pot?'#d3dce8':'#e4e9f1',5);
      if(state.domain==='kitchen'&&highlighted.has(id)) {
        this.rect(ctx,x+3,y+3,c-6,c-6,'#fff0b8',5);
        ctx.strokeStyle='#b36b00';ctx.lineWidth=3;ctx.strokeRect(x+3,y+3,c-6,c-6);
      }
      if (charger) this.label(ctx,'⚡',x+c/2,y+c/2,c*.45,'white');
      else if (pot) { ctx.strokeStyle='#5f6e82';ctx.lineWidth=4;ctx.beginPath();ctx.arc(x+c/2,y+c*.45,c*.25,0,Math.PI*2);ctx.stroke(); }
      const ingredient = station.ingredient || ({eggs:'egg',egg:'egg',tomato:'tomato',meat:'meat',pepper:'pepper'}[id]) || (id.startsWith('ingredients_') ? id.slice(12) : id.startsWith('ingredient_') ? id.slice(11) : null);
      if (ingredient) this.item(ctx,{ingredient},x+c/2,y+c*.42,c*.55);
      if(id==='trash') {
        this.rect(ctx,x+c*.34,y+c*.28,c*.32,c*.32,'#67758a',3);
        this.rect(ctx,x+c*.29,y+c*.22,c*.42,c*.06,'#45546a',2);
        this.label(ctx,'×',x+c*.5,y+c*.45,c*.25,'white');
      }
      const shortLabels={trash:['Trash · E','垃圾桶 · E'],prep:['Prep','备料'],human_buffer:['Your counter','你的暂存台'],plate:['Serving plates','正式餐盘'],serve:['Serve','上菜'],handoff:['Handoff','交接'],protein1:['Temp. plate 1','熟料临时盘 1'],protein2:['Temp. plate 2','熟料临时盘 2'],ai_raw:['2 raw slots','原料双槽']};
      const text = shortLabels[id]?.[this.lang==='zh'?1:0] || (this.lang==='zh' ? station.label_zh : station.label_en);
      if (!charger) this.label(ctx,text || id,x+c/2,y+c*.82,Math.min(11,c*.145),'#536279');
    }
    if (state.domain === 'warehouse') {
      for (const charger of state.chargers || []) {
        const [x,y] = at(Array.isArray(charger)?charger[0]:charger.x,Array.isArray(charger)?charger[1]:charger.y);
        this.rect(ctx,x+4,y+4,c-8,c-8,'#6558e8',2); this.label(ctx,'⚡',x+c/2,y+c/2,c*.46,'white');
      }
      (state.orders || []).forEach(order => {
        const {slot,color}=warehouseOrders.get(order.id);
        for (const [key,letter] of [['pickup','A'],['dropoff','B']]) {
          const pos=order[key]; if(!pos || (key==='pickup' && order.status!=='available' && order.status!=='waiting')) continue;
          const [x,y]=at(Array.isArray(pos)?pos[0]:pos.x,Array.isArray(pos)?pos[1]:pos.y);
          ctx.fillStyle=color;ctx.beginPath();ctx.arc(x+c/2,y+c/2,c*.31,0,Math.PI*2);ctx.fill();
          this.label(ctx,letter+slot,x+c/2,y+c/2,c*.27,'white',800);
        }
      });
    } else {
      for (const pot of state.pots || []) {
        const [x,y]=at(pot.x,pot.y);
        if (pot.status !== 'empty') { this.item(ctx,pot.item || {ingredient:pot.ingredient,recipe:pot.recipe,stage:pot.phase},x+c/2,y+c*.38,c*.52); this.label(ctx,pot.status==='burnt'?'×':pot.status==='cooking'?pot.remaining:'✓',x+c/2,y+c*.64,c*.19,pot.status==='burnt'?'#d9485f':'#26324a'); }
      }
      if (state.handoff) { const station=stations.find(s=>s.id==='handoff'); if(station){const [x,y]=at(station.x,station.y);this.item(ctx,state.handoff,x+c/2,y+c*.38,c*.52);this.freshnessBadge(ctx,state.handoff,state,x+c/2,y+c*.64,c*.86,c*.19,true);} }
      const buffers=state.buffers||{};
      const place=(id,item,dx=0)=>{const st=stations.find(s=>s.id===id);if(st&&item){const [x,y]=at(st.x,st.y);this.item(ctx,item,x+c/2+dx*c,y+c*.37,c*.43);this.freshnessBadge(ctx,item,state,x+c/2+dx*c,y+c*.64,c*(id==='ai_raw'?.43:.86),c*.19,true);}};
      place('human_buffer',buffers.human);
      (buffers.ai_raw||[]).forEach((item,i)=>place('ai_raw',item,i? .22:-.22));
      for(const [pot,item] of Object.entries(buffers.protein||{})) { const st=stations.find(s=>s.kind==='protein_buffer'&&s.pot_id===pot);if(st&&item){const [x,y]=at(st.x,st.y);this.item(ctx,item,x+c/2,y+c*.38,c*.5);} }
    }
    for (const [actor,color] of [['human','#4f6ff0'],['ai','#f56b3d']]) {
      const a=after[actor],b=before[actor]||a;if(!a)continue;
      const [xpos,ypos]=this.actorPosition(before,after,actor,p);
      // Warehouse facts and A/B slot numbers switch with the displayed map,
      // so a replenished job cannot relabel cargo halfway through a move.
      const [x,y]=at(xpos,ypos),visible=state.domain==='warehouse'?state[actor]:(p<.5?b:a);
      this.canvas.dataset[actor+'X']=xpos.toFixed(3);this.canvas.dataset[actor+'Y']=ypos.toFixed(3);
      this.rect(ctx,x+c*.14,y+c*.14,c*.72,c*.72,a.active===false?'#d9485f':color,c*.15);
      this.label(ctx,actor==='human'?'1':'2',x+c/2,y+c/2,c*.34,'white',900);
      if(after.domain==='warehouse'&&after.collision_animation&&p>.2&&p<.85) {
        ctx.strokeStyle='#d82c43';ctx.lineWidth=3;
        ctx.strokeRect(x+c*.1,y+c*.1,c*.8,c*.8);
        this.label(ctx,this.lang==='zh'?'碰撞':'Collision',x+c/2,y+c*.94,c*.14,'#b4233b',850);
      }
      if (visible.facing) {
        const [dx,dy]=({up:[0,-1],down:[0,1],left:[-1,0],right:[1,0]})[visible.facing]||[0,1];
        const cx=x+c/2,cy=y+c/2,tipx=cx+dx*c*.5,tipy=cy+dy*c*.5;
        for(const [stroke,lineWidth] of [['#ffffff',c*.13],['#26324a',c*.065]]) {
          ctx.strokeStyle=stroke;ctx.lineWidth=lineWidth;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();
          ctx.moveTo(cx+dx*c*.28,cy+dy*c*.28);ctx.lineTo(tipx,tipy);
          ctx.moveTo(tipx-dx*c*.12-dy*c*.1,tipy-dy*c*.12+dx*c*.1);ctx.lineTo(tipx,tipy);
          ctx.lineTo(tipx-dx*c*.12+dy*c*.1,tipy-dy*c*.12-dx*c*.1);ctx.stroke();
        }
      }
      if (visible.battery!==undefined) {
        const top=Math.max(oy+2,y-c*.12);
        this.rect(ctx,x+c*.14,top,c*.49,c*.22,'#fff',8);
        this.label(ctx,Math.round(visible.battery)+'%',x+c*.385,top+c*.11,c*.16,visible.battery<=20?'#b4233b':'#26324a',800);
      }
      if (state.domain==='warehouse' && visible.carrying) {
        const cargo=warehouseOrders.get(visible.carrying),cx=x+c*.85,cy=y+c*.15;
        ctx.fillStyle=cargo?.color||'#b79f00';ctx.beginPath();ctx.arc(cx,cy,c*.17,0,Math.PI*2);ctx.fill();
        this.label(ctx,cargo?'A'+cargo.slot:'?',cx,cy,c*.18,'white',800);
      } else if (state.domain!=='warehouse') {
        this.item(ctx,visible.holding,x+c*.86,y+c*.15,c*.43);
        if(actor==='human'&&visible.preparation&&!this.freshnessLabel(visible.holding,p<.5?before:after)) {
          const prep=visible.preparation;
          const verb=prep.ingredient==='egg'?(this.lang==='zh'?'打散':'Whisk'):(this.lang==='zh'?'切配':'Chop');
          const text=prep.ready?(this.lang==='zh'?'✓ 已备好':'✓ Prepared'):(this.lang==='zh'?`${verb}还需 ${prep.remaining} 次`:`${verb}: ${prep.remaining} more`);
          this.rect(ctx,x+c*.02,y+c*.88,c*.96,c*.21,prep.ready?'#dcf5e8':'#fff2d1',3);
          this.label(ctx,text,x+c*.5,y+c*.985,c*.125,prep.ready?'#16704c':'#785f21',750);
        }
        this.freshnessBadge(ctx,visible.holding,p<.5?before:after,x+c*.5,y+c*.985,c*.96,c*.21);
        if(actor==='ai') {
          const recipes=[...new Map((visible.current_cooking||[]).map(job=>[job.recipe,job])).values()];
          recipes.slice(0,2).forEach((job,i)=>{
            const text=this.lang==='zh'?job.recipe_label_zh:job.recipe_label_en;
            const top=Math.max(oy+1,y-c*(.23*(recipes.length-i)));
            this.rect(ctx,x-c*.22,top,c*1.44,c*.21,'#fff5e9',3);
            this.label(ctx,text,x+c*.5,top+c*.105,this.lang==='zh'?c*.145:c*.105,'#9b421f',750);
          });
        }
      }
    }
  }
  pong(ctx,before,after,p,w,h) {
    const lane=x=>38+x*58, line=585, top=65, y=value=>top+value/12*(line-top);
    ctx.strokeStyle='#dce4ef';ctx.lineWidth=1;
    for(let col=0;col<after.lanes;col++){ctx.beginPath();ctx.moveTo(lane(col),top);ctx.lineTo(lane(col),line);ctx.stroke();this.label(ctx,col+1,lane(col),610,12,'#68758b');}
    for(let row=0;row<=12;row++){ctx.beginPath();ctx.moveTo(16,y(row));ctx.lineTo(w-16,y(row));ctx.stroke();}
    const next=new Map(after.balls.map(b=>[b.id,b]));
    for(const b of before.balls){
      const target=next.get(b.id),start=b.y??12-b.remaining*(b.vy||(b.kind==='ordinary'?2:1));
      const end=target ? (target.y??12-target.remaining*(target.vy||(target.kind==='ordinary'?2:1))) : 12;
      if(!target&&p===1)continue;
      this.ball(ctx,p===1?target:b,lane,y(start+(end-start)*p),p===1?after.balls:before.balls);
    }
    if(p===1)for(const b of after.balls)if(!before.balls.some(old=>old.id===b.id))this.ball(ctx,b,lane,y(b.y??12-b.remaining*(b.vy||(b.kind==='ordinary'?2:1))),after.balls);
    for(const [actor,color]of[['human','#4f6ff0'],['ai','#f56b3d']]){
      const x=before[actor].x+(after[actor].x-before[actor].x)*p;
      const overlap=Math.abs(after.human.x-after.ai.x)<.01;
      this.rect(ctx,lane(x)-22,line+(actor==='ai'&&overlap?15:0)-7,44,13,color,5);
      this.canvas.dataset[actor+'X']=x.toFixed(3);
    }
    this.label(ctx,this.lang==='zh'?'小球 2格/步 · 大球 1格/步':'Small: 2 cells / move · Team: 1 cell / move',w/2,18,13,'#68758b');
  }
  ball(ctx,b,lane,y,balls=[]) {
    const label=b.id;
    if(b.contacts.length===2){
      // Simultaneous team balls share their true height. Separate their narrow
      // drawing bands so overlapping spans cannot hide one another.
      const peers=balls.filter(other=>other.contacts.length===2&&other.remaining===b.remaining).sort((a,b)=>a.id.localeCompare(b.id));
      const index=Math.max(0,peers.findIndex(other=>other.id===b.id));
      const band=y+(peers.length>1?(index-(peers.length-1)/2)*15:0);
      const x1=lane(b.contacts[0]),x2=lane(b.contacts[1]);
      this.rect(ctx,x1-10,band-6,x2-x1+20,12,index%2?'#9e7929':'#c1a447',6);
      for(const x of[x1,x2]){ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(x,band,4,0,Math.PI*2);ctx.fill();}
      this.label(ctx,label,(x1+x2)/2,band+(peers.length>1&&index>0?15:-15),10,'#806626');
    }
    else{ctx.fillStyle='#697e9e';ctx.beginPath();ctx.arc(lane(b.contacts[0]),y,10,0,Math.PI*2);ctx.fill();this.label(ctx,label,lane(b.contacts[0]),y-20,10,'#536279');}
  }
};
