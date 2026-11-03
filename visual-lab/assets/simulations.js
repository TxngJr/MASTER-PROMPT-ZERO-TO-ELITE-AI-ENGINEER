/* Deterministic teaching models. No network, trained weights or external dependencies. */
(function () {
  const C={ink:'#182e35',muted:'#62767c',mint:'#087f73',lime:'#bde8aa',blue:'#4b72cc',orange:'#d47a39',red:'#c25059',line:'#dbe5e2',pale:'#e8f4ef'};
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const f=(n,d=3)=>Number.isFinite(n)?Number(n.toFixed(d)).toString():'∞';
  const text=(x,y,t,size=14,color=C.ink,anchor='start')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${color}" text-anchor="${anchor}">${esc(t)}</text>`;
  const rect=(x,y,w,h,color,rx=8,extra='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${color}" ${extra}/>`;
  const line=(x,y,x2,y2,color=C.line,w=2,extra='')=>`<line x1="${x}" y1="${y}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}" ${extra}/>`;
  const circle=(x,y,r,color,extra='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" ${extra}/>`;
  const box=(x,y,w,h,title,value,color=C.mint)=>rect(x,y,w,h,C.pale)+text(x+16,y+25,title,12,C.muted)+text(x+16,y+57,value,24,color);
  const softmax=a=>{const m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)};
  const sum=a=>a.reduce((a,b)=>a+b,0);
  const sigmoid=x=>1/(1+Math.exp(-x));
  const path=(points,color=C.mint,width=3,extra='')=>`<polyline points="${points.map(v=>v.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" ${extra}/>`;
  const bars=(labels,values,x=80,y=72,w=640,h=170,colors=[])=>values.map((v,i)=>{const bh=Math.max(0,v)*h,bw=w/values.length-18,bx=x+i*w/values.length;return rect(bx,y+h-bh,bw,bh,colors[i]||C.mint,5)+text(bx+bw/2,y+h+24,labels[i],12,C.muted,'middle')+text(bx+bw/2,y+h-bh-10,f(v),13,C.ink,'middle')}).join('');
  function result(body,metrics,note,code,values={}) {return {body,metrics,note,code,values};}
  const points=[[1.0,1.0,0],[1.6,2.3,0],[2.2,1.5,0],[3.2,3.1,1],[3.7,1.9,0],[4.3,3.8,1],[5.3,2.5,1],[6.1,4.1,1],[6.8,3.2,1]];
  const gx=x=>65+x*57,gy=y=>278-y*48;
  function scatter(pts=points) {return line(50,278,505,278)+line(50,278,50,36)+text(508,294,'x₁',12,C.muted)+text(29,35,'x₂',12,C.muted)+pts.map(p=>circle(gx(p[0]),gy(p[1]),7,p[2]?C.orange:C.blue)).join('');}
  function simulate(lesson,p,stage=3) {
    const id=lesson.id,e=lesson.engine;
    const show=stage>=2;
    if(e==='program') {
      const xs=[2,4,6,8],picked=xs.filter(x=>x>p),total=sum(picked);
      const body=xs.map((x,i)=>{const yes=x>p,bx=65+i*190;return box(bx,44,150,80,'input',x,C.ink)+line(bx+75,130,bx+75,177,yes?C.mint:C.line,3)+rect(bx,180,150,54,yes?C.pale:'#f1f1ec')+text(bx+75,213,`${x} > ${p} : ${yes?'True':'False'}`,15,yes?C.mint:C.muted,'middle')}).join('')+box(300,264,300,72,'return (after filter)',show?total:'?',C.mint);
      return result(body,[['input',xs.join(', ')],['selected',picked.join(', ')||'ไม่มี'],['sum',total]],'เส้นสีเขียวคือ branch ที่ผ่านเงื่อนไข >; ค่าที่เท่ากับเกณฑ์ถูกตัดออก',`xs = [2, 4, 6, 8]\nthreshold = ${p}\npicked = [x for x in xs if x > threshold]\ntotal = sum(picked)  # ${total}`,{total,picked});
    }
    if(e==='math') {
      const sx=x=>280+x*65,sy=y=>282-y*24,curve=[];
      for(let x=-3;x<=3;x+=0.1)curve.push([sx(x),sy(x*x)]);
      const a=Math.max(-3,p-1),b=Math.min(3,p+1),y=x=>p*p+2*p*(x-p);
      return result(line(65,282,500,282)+line(280,30,280,300)+path(curve)+path([[sx(a),sy(y(a))],[sx(b),sy(y(b))]],C.orange)+circle(sx(p),sy(p*p),7,C.mint)+box(568,55,245,80,'f(x)',f(p*p))+box(568,157,245,80,'df/dx',f(2*p),C.orange),[['x',p],['f(x)',f(p*p)],['gradient',f(2*p)]],'เส้นสีส้มคือ tangent ณ จุดปัจจุบัน; slope ติดลบทางซ้าย และบวกทางขวา',`x = ${p}\nf = x ** 2       # ${f(p*p)}\ngrad = 2 * x     # ${f(2*p)}`,{gradient:2*p});
    }
    if(e==='array') {
      const bias=[1,2,3],rows=Array.from({length:p},(_,i)=>[i+1,i+2,i+3]),means=bias.map((b,j)=>sum(rows.map(r=>r[j]+b))/p);
      let body=text(65,35,`X (${p}, 3)`,17)+text(380,35,'bias (3,)',17)+text(620,35,`Y (${p}, 3)`,17);
      rows.forEach((row,i)=>row.forEach((x,j)=>{body+=rect(65+j*70,55+i*35,62,29,'#edf1f8')+text(96+j*70,76+i*35,x,15,C.blue,'middle')+rect(620+j*70,55+i*35,62,29,C.pale)+text(651+j*70,76+i*35,x+bias[j],15,C.mint,'middle')}));
      bias.forEach((b,j)=>{body+=rect(380+j*70,55,62,29,'#faeddf')+text(411+j*70,76,b,15,C.orange,'middle')});
      body+=text(390,160,'broadcast ↓ B rows',13,C.muted)+box(330,279,400,65,'mean(axis=0) → shape (3,)',means.map(v=>f(v,1)).join('  '));
      return result(body,[['batch B',p],['elements',p*3],['mean shape','(3,)']],'เพิ่ม B เปลี่ยนจำนวนแถว แต่ bias และ feature means ยังคงมี 3 ช่อง',`X = np.array([[i+1, i+2, i+3] for i in range(${p})])\nY = X + np.array([1, 2, 3])\nfeature_means = Y.mean(axis=0)`,{means});
    }
    if(e==='data') {
      const records=['A','B','C','C','D','E','F','G','H','I'],unique=[...new Set(records)],n=Math.floor(unique.length*p/100),train=unique.slice(0,n),test=unique.slice(n);
      let body=text(55,35,'RAW RECORDS',13,C.muted)+records.map((r,i)=>rect(55+i*76,53,65,40,r==='C'&&i===3?'#fce8e8':'#edf1f8')+text(87+i*76,79,r,16,C.ink,'middle')).join('');
      body+=text(55,129,'DEDUP → SPLIT → FIT ON TRAIN',13,C.muted)+rect(55,148,470,103,C.pale)+text(75,175,`TRAIN · ${train.length} records`,14,C.mint)+text(75,218,train.join('   '),25)+rect(550,148,285,103,'#faeddf')+text(570,175,`TEST · ${test.length} records`,14,C.orange)+text(570,218,test.join('   '),25)+text(55,302,'μ, σ มาจาก TRAIN แล้วส่งให้ TEST ใช้ transform',16,C.mint);
      return result(body,[['duplicates removed',1],['train',n],['test',9-n]],'ตัวอย่าง 10 records เหลือ 9 หลัง dedup; ใช้ floor เมื่อแบ่งตามสัดส่วน จึงไม่ตรงเปอร์เซ็นต์เป๊ะ',`unique = deduplicate(records)\ntrain, test = split(unique, ratio=${p/100})\nscaler.fit(train)\ntrain_z = scaler.transform(train)\ntest_z = scaler.transform(test)`,{train,test});
    }
    if(e==='train') {
      const xs=[0.5,1,1.5,2],ys=xs.map(x=>2*x),w0=0.4,grad=w=>2*sum(xs.map((x,i)=>x*(w*x-ys[i])))/xs.length,loss=w=>sum(xs.map((x,i)=>(w*x-ys[i])**2))/xs.length;
      let ws=[w0];for(let i=0;i<6;i++)ws.push(ws.at(-1)-p*grad(ws.at(-1)));
      const w=ws[Math.min(stage,3)],l=loss(w),history=ws.map(loss),maxL=Math.max(10,...history),chartY=v=>270-180*v/maxL;
      let body=line(55,270,405,270)+line(55,270,55,50)+text(65,35,'LOSS TRAJECTORY · 6 STEPS',13,C.muted)+path(history.map((v,i)=>[65+i*50,chartY(v)]));
      history.forEach((v,i)=>{body+=circle(65+i*50,chartY(v),i===stage?7:4,i===stage?C.orange:C.mint)});
      body+=text(55,299,'0',12,C.muted)+text(355,299,'6 updates',12,C.muted)+box(475,50,330,78,'weight at highlighted step',f(w))+box(475,148,330,78,'MSE',f(l),C.orange)+text(475,270,'target slope = 2 · b = 0',16,C.muted);
      return result(body,[['η',p],['initial gradient',f(grad(w0))],['after 6 steps loss',f(history.at(-1))]],'เส้นนี้คำนวณจาก regression 4 จุดจริง; บท LLM/fine-tuning ใช้ภาพนี้เป็นตัวแทน optimizer step ไม่ใช่ loss ของ LLM',`x = np.array([0.5, 1.0, 1.5, 2.0]); y = 2 * x\nw = ${f(w)}\ng = 2 * np.mean(x * (w*x - y))  # ${f(grad(w))}\nw_next = w - ${p} * g  # ${f(w-p*grad(w))}`,{history,ws,gradient:grad(w0)});
    }
    if(e==='sigmoid'||e==='activation'||e==='gan') {
      const tau=e==='sigmoid'?p:0.5,z=e==='sigmoid'?1:p,prob=sigmoid(z),loss=Math.max(z,0)-z+Math.log1p(Math.exp(-Math.abs(z))),g=prob-1;
      const sx=v=>65+(v+6)*35,sy=v=>268-v*200,curve=[];for(let x=-6;x<=6;x+=0.1)curve.push([sx(x),sy(sigmoid(x))]);
      const body=line(60,268,510,268)+line(65,268,65,52)+path(curve)+line(65,sy(tau),485,sy(tau),C.orange,2,'stroke-dasharray="5 5"')+circle(sx(z),sy(prob),7,C.mint)+text(65,300,'−6           logit z           +6',14,C.muted)+box(565,48,270,80,'σ(z)',f(prob))+box(565,151,270,80,e==='sigmoid'?'decision class':'loss (target=1)',e==='sigmoid'?(prob>=tau?1:0):f(loss),C.orange)+text(565,275,e==='sigmoid'?`threshold τ = ${p}`:`dL/dz = ${f(g)}`,17);
      return result(body,[['logit',z],['probability',f(prob)],e==='sigmoid'?['class',prob>=tau?1:0]:['gradient',f(g)]],e==='gan'?'นี่คือ loss ของ generator จาก logit ที่ D ให้ fake; ไม่ได้รัน GAN ที่ฝึกจริง':'sigmoid เปลี่ยน score เป็น probability; การขยับ threshold เปลี่ยนคำตัดสินโดยไม่เปลี่ยน probability',`z = ${z}\np = 1 / (1 + exp(-z))  # ${f(prob)}\n${e==='sigmoid'?`class_id = int(p >= ${p})`:`loss = softplus(-z)  # ${f(loss)}\ngrad_z = p - 1       # ${f(g)}`}`,{prob,loss,gradient:g});
    }
    if(e==='geometry') {
      let body=scatter(),metrics,code,note,values={};
      if(id===8||id===45) {
        const q=[3.8,2.9],k=p,sorted=points.map((a,i)=>({i,d:Math.hypot(a[0]-q[0],a[1]-q[1]),class:a[2]})).sort((a,b)=>a.d-b.d),chosen=sorted.slice(0,k),votes=sum(chosen.map(a=>a.class)),radius=chosen.at(-1).d;
        body+=`<ellipse cx="${gx(q[0])}" cy="${gy(q[1])}" rx="${radius*57}" ry="${radius*48}" fill="none" stroke="${C.mint}" stroke-dasharray="6 5"/>`+circle(gx(q[0]),gy(q[1]),9,C.mint);
        chosen.forEach(a=>{body+=circle(gx(points[a.i][0]),gy(points[a.i][1]),12,'none',`stroke="${C.mint}" stroke-width="2"`)});
        body+=box(550,55,285,80,'selected neighbors',k)+box(550,158,285,80,id===8?'class vote 0 : 1':'k-th distance',id===8?`${k-votes} : ${votes}`:f(radius));
        metrics=[['query','(3.8, 2.9)'],['nearest distance',f(sorted[0].d)],['top-k',k]];values={selected:chosen.map(a=>a.i),radius};
        note='จุดเขียวคือ query; วงรีในภาพเกิดจาก scale ของแกน แต่คำนวณระยะ Euclidean ในพิกัดข้อมูล';code=`distances = sqrt(((X - q)**2).sum(axis=1))\nindices = argsort(distances)[:${k}]\n# distances: ${chosen.map(a=>f(a.d)).join(', ')}`;
      } else if(id===13) {
        const mid=3.7,width=1/p;body+=line(gx(mid),40,gx(mid),278,C.mint,3)+line(gx(mid-width),40,gx(mid-width),278,C.orange,2,'stroke-dasharray="5 5"')+line(gx(mid+width),40,gx(mid+width),278,C.orange,2,'stroke-dasharray="5 5"')+box(550,55,285,80,'margin width',f(2/p))+box(550,158,285,80,'||w||',p);
        metrics=[['||w||',p],['margin',f(2/p)],['boundary x₁',mid]];note='เส้นกลาง x₁=3.7 คงที่; ปรับ norm โดยเปลี่ยน b คู่กันเพื่อสาธิต canonical margin';code=`w = [${p}, 0]\nb = -3.7 * ${p}\nmargin = 2 / norm(w)  # ${f(2/p)}`;values={margin:2/p};
      } else if(id===14) {
        let mus=[[1,4],[6,1]],assigned=[];for(let t=0;t<=p;t++){assigned=points.map(a=>Math.hypot(a[0]-mus[0][0],a[1]-mus[0][1])<=Math.hypot(a[0]-mus[1][0],a[1]-mus[1][1])?0:1);if(t<p)mus=mus.map((m,k)=>{const ps=points.filter((a,i)=>assigned[i]===k);return ps.length?[sum(ps.map(a=>a[0]))/ps.length,sum(ps.map(a=>a[1]))/ps.length]:m})}
        const inertia=sum(points.map((a,i)=>(a[0]-mus[assigned[i]][0])**2+(a[1]-mus[assigned[i]][1])**2));body=scatter(points.map((a,i)=>[a[0],a[1],assigned[i]]));mus.forEach((m,k)=>{body+=rect(gx(m[0])-9,gy(m[1])-9,18,18,k?C.orange:C.blue,2,'stroke="white" stroke-width="3"')});body+=box(550,55,285,80,'Lloyd iterations',p)+box(550,158,285,80,'inertia J',f(inertia));
        metrics=[['clusters',2],['iteration',p],['inertia',f(inertia)]];code=`labels = argmin(distance(X, centers), axis=1)\ncenters = group_mean(X, labels)\n# centers: ${mus.map(m=>m.map(v=>f(v)).join(',')).join(' / ')}`;note='สี่เหลี่ยมคือ centroid; อัปเดตด้วยค่าเฉลี่ยจริงของสมาชิกในแต่ละกลุ่ม';values={mus,inertia};
      } else if(id===15) {
        const a=p*Math.PI/180,u=[Math.cos(a),Math.sin(a)],center=[sum(points.map(v=>v[0]))/points.length,sum(points.map(v=>v[1]))/points.length],zs=points.map(v=>(v[0]-center[0])*u[0]+(v[1]-center[1])*u[1]),mean=sum(zs)/zs.length,variance=sum(zs.map(z=>(z-mean)**2))/zs.length;
        body+=line(gx(center[0]-3*u[0]),gy(center[1]-3*u[1]),gx(center[0]+3*u[0]),gy(center[1]+3*u[1]),C.mint,3);
        points.forEach((v,i)=>{const projected=[center[0]+zs[i]*u[0],center[1]+zs[i]*u[1]];body+=line(gx(v[0]),gy(v[1]),gx(projected[0]),gy(projected[1]),C.muted,1,'stroke-dasharray="3 3"')+circle(gx(projected[0]),gy(projected[1]),4,C.mint)});
        body+=box(550,55,285,80,'projection angle',`${p}°`)+box(550,158,285,80,'population variance(z)',f(variance));metrics=[['u',u.map(v=>f(v)).join(', ')],['variance',f(variance)],['dimensions','2 → 1']];note='ภาพแสดงการฉายบนทิศที่เลือก; การหา principal component คือค้นทิศที่ variance สูงสุด';code=`u = [cos(${p}°), sin(${p}°)]\nz = (X - X.mean(axis=0)) @ u\nvariance = np.var(z)`;values={variance};
      } else if(id===16) {
        const xs=[-2.8,-1.3,-0.8,-0.3,0.1,0.4,0.9,1.4,2.7],flags=xs.map(x=>Math.abs(x)>p);body=line(70,185,505,185)+rect(285-p*65,90,p*130,150,C.pale)+text(100,55,'REFERENCE μ=0, σ=1',15,C.muted)+xs.map((x,i)=>circle(285+x*65,160+(i%3)*18,8,flags[i]?C.red:C.mint)).join('')+box(550,55,285,80,'threshold |z|',p)+box(550,158,285,80,'alerts',flags.filter(Boolean).length);
        metrics=[['threshold',p],['flagged',flags.filter(Boolean).length],['reference','μ=0, σ=1']];note='ในตัวอย่างใช้ μ=0, σ=1 เป็น reference คงที่; ไม่ประมาณสถิติใหม่จากจุดที่แสดง';code=`z = (x - 0) / 1\nalerts = abs(z) > ${p}`;values={count:flags.filter(Boolean).length};
      } else {
        const q=[Math.cos(p*Math.PI/180),Math.sin(p*Math.PI/180)],vs=[[1,0],[0.8,0.6],[-0.2,0.98]],scores=vs.map(v=>(q[0]*v[0]+q[1]*v[1])/Math.hypot(...v));body=line(265,90,265,270)+line(85,230,480,230)+line(265,230,265+q[0]*130,230-q[1]*130,C.mint,4)+vs.map((v,i)=>line(265,230,265+v[0]*120,230-v[1]*120,i===0?C.blue:C.orange,3)+text(265+v[0]*140,230-v[1]*140,`word ${i+1}`,13)).join('')+box(550,55,285,80,'query angle',`${p}°`)+box(550,158,285,80,'cosines',scores.map(v=>f(v,2)).join(' / '));metrics=[['angle',p],['best match',scores.indexOf(Math.max(...scores))+1],['norm(q)',1]];note='vectors ตัวอย่างกำหนดเพื่อสอน cosine; ไม่ได้อ้างว่าเป็น embeddings จาก corpus จริง';code=`q = [cos(${p}°), sin(${p}°)]\nscores = dot(q, V.T) / norm(V, axis=1)`;values={scores};
      }
      return result(body,metrics,note,code,values);
    }
    if(e==='bayes') {
      const scores=[(1-p)*0.2,p*0.8],post=scores.map(v=>v/sum(scores));return result(bars(['class 0','class 1'],post,100,80,400,175,[C.blue,C.orange])+box(570,55,265,80,'likelihood P(e | 0,1)','0.2 / 0.8')+box(570,160,265,80,'posterior class 1',f(post[1])),[['prior class 1',p],['normalizer',f(sum(scores))],['posterior 1',f(post[1])]],'likelihood คงที่; เปลี่ยน prior แล้ว normalize ใหม่จึงเห็นผล base rate',`score_0 = (1-${p}) * 0.2\nscore_1 = ${p} * 0.8\nposterior_1 = score_1 / (score_0 + score_1)`,{post});
    }
    if(e==='tree') {
      const labels=[0,0,0,1,0,1,1,1,1,1],left=labels.slice(0,p),right=labels.slice(p),gini=a=>a.length?1-(sum(a)/a.length)**2-(1-sum(a)/a.length)**2:0,g=gini(labels)-(left.length*gini(left)+right.length*gini(right))/10;
      const body=box(315,25,260,70,'split',`x ≤ ${p}?`)+line(445,99,225,165,C.blue,3)+line(445,99,665,165,C.orange,3)+box(80,175,300,90,`LEFT · n=${left.length}`,`Gini ${f(gini(left))}`,C.blue)+box(510,175,300,90,`RIGHT · n=${right.length}`,`Gini ${f(gini(right))}`,C.orange)+text(225,300,left.join(' '),17,C.blue,'middle')+text(665,300,right.join(' '),17,C.orange,'middle');
      return result(body,[['parent Gini',f(gini(labels))],['gain',f(g)],['child sizes',`${p} / ${10-p}`]],'10 จุดเรียงตาม x=1…10; split คำนวณ class proportions ของลูกจริง',`left = y[x <= ${p}]\nright = y[x > ${p}]\ngain = gini(y) - len(left)/10*gini(left) - len(right)/10*gini(right)`,{gain:g});
    }
    if(e==='forest') {
      const probs=[0.2,0.8,0.65,0.4,0.7,0.9,0.3,0.6,0.8].slice(0,p),mean=sum(probs)/p;return result(bars(probs.map((v,i)=>`T${i+1}`),probs,65,65,520,170)+box(625,65,215,80,'mean P(class 1)',f(mean))+box(625,180,215,80,'decision (τ=.5)',mean>=0.5?1:0),[['trees',p],['mean',f(mean)],['range',`${Math.min(...probs)}–${Math.max(...probs)}`]],'tree scores จำลองเพื่อแสดง aggregation; ไม่ได้ fit trees หรือประมาณ uncertainty ของ forest',`probs = [${probs.join(', ')}]\np_mean = sum(probs)/${p}  # ${f(mean)}\nclass_id = int(p_mean >= 0.5)`,{mean});
    }
    if(e==='boost') {
      let pred=[0];for(let i=0;i<6;i++)pred.push(pred.at(-1)+p*0.5*(4-pred.at(-1)));
      return result(line(70,55,490,55,C.orange,2,'stroke-dasharray="4 4"')+text(505,60,'target 4',13,C.orange)+path(pred.map((v,i)=>[80+i*65,278-v*55]))+pred.map((v,i)=>circle(80+i*65,278-v*55,6,C.mint)+text(80+i*65,310,`m${i}`,12,C.muted,'middle')).join('')+box(605,70,235,80,'F₆',f(pred.at(-1)))+box(605,175,235,80,'residual',f(4-pred.at(-1)),C.orange),[['η',p],['weak learner','½ residual'],['residual after 6',f(4-pred.at(-1))]],'ตัวอย่าง scalar: learner แต่ละรอบแก้ครึ่ง residual ก่อนคูณ learning rate',`F = 0; target = 4\nfor m in range(6):\n    h = 0.5 * (target - F)\n    F += ${p} * h`,{pred});
    }
    if(e==='network') {
      const z=2*p+0.5,h=Math.max(0,z),pred=1.5*h,l=(pred-3)**2,g=2*(pred-3)*1.5*(z>0?1:0)*2;
      const xs=[48,245,445,650],titles=['x=2','z=wx+0.5','ReLU(z)','ŷ=1.5h'],vs=[2,z,h,pred];let body='';xs.forEach((x,i)=>{if(i<3)body+=line(x+145,100,x+190,100,id===19?C.orange:C.mint,3,'marker-end="url(#arrow)"');body+=box(x,55,145,80,titles[i],f(vs[i]),i===stage?C.orange:C.mint)});body+=box(125,205,260,80,'loss (target=3)',f(l),C.orange)+box(450,205,330,80,'dL/dw (chain rule)',f(g));
      return result(body,[['z',f(z)],['prediction',f(pred)],['dL/dw',f(g)]],'ตัวเลขจาก neuron scalar ที่คำนวณจริง; gradient ผ่าน ReLU เป็น 0 เมื่อ z<0 (ที่ z=0 ใช้ convention 0)',`x=2; b=0.5; v=1.5; y=3\nz=x*${p}+b; h=max(0,z); pred=v*h\nloss=(pred-y)**2\ngrad_w=2*(pred-y)*v*(z>0)*x  # ${f(g)}`,{z,h,pred,loss:l,gradient:g});
    }
    if(e==='conv') {
      const image=Array.from({length:5},()=>[0,0,1,1,1]),kernel=[[-1,0,1],[-1,0,1],[-1,0,1]],out=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>sum(kernel.flatMap((r,a)=>r.map((v,b)=>v*image[i+a][j+b]))))),row=Math.floor(p/3),col=p%3;
      let body=text(55,35,'INPUT 5×5',14,C.muted)+text(380,35,'KERNEL 3×3',14,C.muted)+text(645,35,'OUTPUT 3×3',14,C.muted);
      image.forEach((r,i)=>r.forEach((v,j)=>{body+=rect(55+j*42,60+i*42,38,38,v?'#accfc2':'#e9eeeb',2)+text(74+j*42,85+i*42,v,14,C.ink,'middle')}));body+=`<rect x="${53+col*42}" y="${58+row*42}" width="126" height="126" fill="none" stroke="${C.orange}" stroke-width="4"/>`;
      kernel.forEach((r,i)=>r.forEach((v,j)=>{body+=rect(380+j*55,65+i*55,48,48,'#faeddf',4)+text(404+j*55,96+i*55,v,16,C.orange,'middle')}));out.forEach((r,i)=>r.forEach((v,j)=>{body+=rect(645+j*55,65+i*55,48,48,i===row&&j===col?C.mint:C.pale,4)+text(669+j*55,96+i*55,stage>=2?v:'?',16,i===row&&j===col?'white':C.mint,'middle')}));
      return result(body,[['position',`(${row},${col})`],['dot sum',out[row][col]],['output shape','3×3']],'kernel นี้ตอบสนองต่อขอบแนวตั้ง; highlight สีส้มเป็น patch ที่กำลังอ่าน',`patch = X[${row}:${row+3}, ${col}:${col+3}]\nY[${row},${col}] = sum(patch * K)  # ${out[row][col]}`,{out,row,col});
    }
    if(e==='sequence'||e==='gates') {
      let vals,labels,note,code;
      if(e==='gates'){vals=Array.from({length:9},(_,i)=>p**i);labels=vals.map((v,i)=>`t${i}`);note='candidate=0 เพื่อแยกผล forget gate; LSTM จริงเพิ่ม input/output gates ด้วย';code=`c0 = 1\nfor t in range(8):\n    c = ${p} * c + 0  # retained memory`}
      else if(id===17){vals=Array.from({length:p+1},(_,i)=>0.8**i);labels=vals.map((v,i)=>`+${i}`);note='AR(1) ตัวอย่างไม่มี intercept หรือ noise; prediction หดเข้าศูนย์และไม่ใช่ forecast จาก dataset จริง';code=`y_now = 1\nforecast = [0.8**h * y_now for h in range(1, ${p+1})]`}
      else {const inputs=[1,-0.5,0.8,0.2];vals=[0];inputs.forEach(x=>vals.push(Math.tanh(x+p*vals.at(-1))));labels=vals.map((v,i)=>`h${i}`);note='tanh และ recurrent weight คำนวณจาก 4 inputs จริง; ค่า hidden อาจติดลบได้';code=`h = 0\nfor x in [1,-0.5,0.8,0.2]:\n    h = tanh(x + ${p} * h)`}
      const body=line(65,200,835,200)+path(vals.map((v,i)=>[80+i*720/(vals.length-1),200-v*125]))+vals.map((v,i)=>circle(80+i*720/(vals.length-1),200-v*125,7,C.mint)+text(80+i*720/(vals.length-1),200-v*125-17,f(v),13,C.ink,'middle')+text(80+i*720/(vals.length-1),280,labels[i],13,C.muted,'middle')).join('');return result(body,[['steps',vals.length-1],['initial',f(vals[0])],['last state',f(vals.at(-1))]],note,code,{vals});
    }
    if(e==='latent') {
      const mu=id===27?0.5:0,sigma=id===27?0.8:1,z=mu+sigma*p,x=2*z;return result(box(50,78,210,88,id===27?'μ, σ':'prior mean, std',`${mu}, ${sigma}`)+line(267,130,330,130,C.mint,3,'marker-end="url(#arrow)"')+box(340,78,210,88,'latent z',f(z))+line(560,130,620,130,C.mint,3,'marker-end="url(#arrow)"')+box(635,78,210,88,'decoder x̂=2z',f(x))+text(450,258,'scalar decoder เพื่อให้คำนวณทุกขั้นได้ด้วยมือ',16,C.muted,'middle'),[['ε / sample',p],['z',f(z)],['output',f(x)]],'ภาพนี้แยก latent sampling กับ decoder; ไม่ได้ฝึก encoder/decoder หรือแสดงคุณภาพของ VAE',`mu=${mu}; sigma=${sigma}; epsilon=${p}\nz=mu+sigma*epsilon  # ${f(z)}\nx_hat=2*z          # ${f(x)}`,{z,x});
    }
    if(e==='attention') {
      const queryMode=[34,35,36].includes(id),causal=[35,72].includes(id),row=queryMode?p-1:3,div=queryMode||id===72?1:p,qs=[[1,0],[0,1],[1,1],[1,-1]],ks=[[1,0],[0,1],[1,1],[-1,1]],vs=[1,2,3,4],q=qs[row],scores=ks.map(k=>(q[0]*k[0]+q[1]*k[1])/Math.SQRT2/div),allowed=ks.map((k,j)=>!causal||(j<=row&&(id!==72||j>=row-p+1))),a=softmax(scores.map((s,j)=>allowed[j]?s:-Infinity)),o=sum(a.map((v,j)=>v*vs[j]));
      let body=text(45,35,`QUERY ${row+1}: [${q.join(', ')}]`,15)+text(330,35,causal?'ALLOWED ATTENTION':stage<2?'KEY / VALUE → DOT SCORE':'SOFTMAX WEIGHTS',14,C.muted);
      for(let i=0;i<4;i++)for(let j=0;j<4;j++){const allow=!causal||(j<=i&&(id!==72||j>=i-p+1));body+=rect(45+j*56,65+i*52,48,44,allow?(i===row?C.mint:C.pale):'#edf0ef',5)+text(69+j*56,92+i*52,allow?'✓':'×',15,i===row&&allow?'white':C.muted,'middle')}
      if(stage<2){body+=ks.map((k,j)=>rect(330,60+j*47,445,39,allowed[j]?C.pale:'#edf0ef')+text(345,85+j*47,`K${j+1}=[${k.join(',')}]  V${j+1}=${vs[j]}  score=${stage===0?'?':allowed[j]?f(scores[j]):'−∞ (masked)'}`,16,allowed[j]?C.mint:C.muted)).join('')}else{body+=bars(vs.map(v=>`V=${v}`),a,330,75,430,145,a.map(v=>v?C.mint:C.line))}
      body+=text(330,270,stage<3?'ขั้นถัดไป: รวม Values ด้วย weights':`weighted sum O = ${f(o)}`,20,stage<3?C.muted:C.mint)+text(330,310,`scores = [${scores.map((v,j)=>allowed[j]?f(v,2):'−∞').join(', ')}]`,13,C.muted);
      return result(body,[['sum weights',f(sum(a))],['output O',f(o)],['keys allowed',allowed.filter(Boolean).length]],'4 tokens, d_k=2; V=[1,2,3,4] เป็น scalar values เพื่ออ่าน weighted sum ได้ ภาพไม่รวม FFN/residual ของ block เต็ม',`scores = q @ K.T / sqrt(2) / ${div}\nscores[masked] = -inf\na = softmax(scores)  # ${a.map(v=>f(v)).join(', ')}\nO = a @ [1,2,3,4]    # ${f(o)}`,{a,o,allowed,scores});
    }
    if(e==='patch') {
      const count=Math.ceil(16/p),n=count**2;let body=text(65,35,'16×16 IMAGE · C=3',14,C.muted);
      for(let i=0;i<count;i++)for(let j=0;j<count;j++)body+=rect(65+j*240/count,55+i*240/count,240/count-3,240/count-3,(i+j)%2?C.mint:'#b9d9bc',3);
      body+=box(400,55,380,75,'patch tokens',`${n} (${count}×${count})`)+box(400,150,380,75,'dense attention pairs (without CLS)',n*n)+text(400,275,`flatten dimension = P²×C = ${p*p*3}`,17);
      return result(body,[['P',p],['tokens',n],['P²C',p*p*3]],'หาก P หารขนาดภาพไม่ลงตัว ภาพใช้ padding ที่ขอบ; token count ใช้ ceil(H/P)ceil(W/P)',`P=${p}; H=W=16; C=3\nN=ceil(H/P)*ceil(W/P)  # ${n}\npatch_dim=P*P*C        # ${p*p*3}`,{n,pairs:n*n});
    }
    if(e==='tokens'||e==='bpe'||e==='sft') {
      let tokens,note,code,metrics;
      if(e==='bpe'){const seqs=[['b','a','n','a','n','a'],['b','an','an','a'],['b','anan','a'],['b','anana']];tokens=seqs[p].map(t=>({t,color:C.mint}));note='character-level BPE ตัวอย่าง corpus banana×2; deterministic tie เลือก (a,n) ก่อน (n,a)';code=`symbols = list('banana')\nmerges = [('a','n'), ('an','an'), ('anan','a')][:${p}]\nencoded = ${JSON.stringify(seqs[p])}\ndecoded = ''.join(encoded)  # banana`;metrics=[['merges',p],['token count',tokens.length],['round trip','banana']];}
      else if(e==='sft'){tokens=[{t:'SYS',color:C.muted},{t:'USER',color:C.blue},...Array.from({length:p},(_,i)=>({t:`A${i+1}`,color:C.mint}))];note='สีเขียวคิด loss; system/user ใช้ ignore labels แต่ยังเป็น context ของ attention';code=`roles = ['system','user', ...assistant_tokens]\nlabels = [-100, -100, ...assistant_ids]\nvalid_count = ${p}\nloss = sum(valid_token_losses) / valid_count`;metrics=[['input tokens',p+2],['valid labels',p],['ignored labels',2]];}
      else{tokens=Array.from('banana').map((t,i)=>({t,color:i<p?C.blue:i===p?C.orange:C.muted}));note='ตัวอักษรสีน้ำเงินเป็น context สีส้มเป็น target ถัดไป; tokenizer จริงต้องกำหนด Unicode policy';code=`tokens = list('banana')\ncontext = tokens[:${p}]  # ${JSON.stringify(Array.from('banana').slice(0,p))}\ntarget = tokens[${p}]    # ${Array.from('banana')[p]}`;metrics=[['context length',p],['target',Array.from('banana')[p]],['sequence length',6]];}
      const bw=Math.min(110,720/tokens.length);let body=tokens.map((a,i)=>rect(65+i*(bw+10),105,bw,85,a.color,8)+text(65+i*(bw+10)+bw/2,154,a.t,23,'white','middle')+text(65+i*(bw+10)+bw/2,220,`pos ${i}`,12,C.muted,'middle')).join('')+text(65,295,e==='sft'?'LOSS MASK: −100, −100, '+Array.from({length:p},()=>1).join(', '):'ลำดับ symbols → context / learned merge ranks → token IDs',16);
      return result(body,metrics,note,code,{count:tokens.length});
    }
    if(e==='graph') {
      const edges=[[0,1],[1,2],[2,3],[3,4],[1,4]],pos=[[100,155],[280,90],[455,155],[640,90],[720,235]],initial=[1,0,0,0,0];let vals=initial;
      for(let t=0;t<p;t++)vals=vals.map((v,i)=>{const ns=edges.flatMap(([a,b])=>a===i?[b]:b===i?[a]:[]);return (v+sum(ns.map(j=>vals[j])))/(ns.length+1)});
      const body=edges.map(([a,b])=>line(...pos[a],...pos[b],C.line,5)).join('')+pos.map(([x,y],i)=>circle(x,y,35,`rgb(${Math.round(216-vals[i]*160)},${Math.round(234-vals[i]*90)},${Math.round(224-vals[i]*100)})`)+text(x,y+5,f(vals[i]),16,C.ink,'middle')+text(x,y+58,`node ${i}`,13,C.muted,'middle')).join('');return result(body,[['rounds',p],['initial signal','node 0 = 1'],['node 4',f(vals[4])]],'รอบอัปเดตใช้ states จากรอบก่อนทั้งหมดพร้อมกัน; mean aggregation รวม self-loop',`h = [1,0,0,0,0]\nfor layer in range(${p}):\n    h_new[i] = mean([h[i]] + h[neighbors(i)])\n    h = h_new`,{vals});
    }
    if(e==='reward') {
      const r=[0,1,3,5],contributions=r.map((v,i)=>v*p**i),ret=sum(contributions);return result(bars(r.map((v,i)=>`t${i} · r=${v}`),contributions.map(v=>v/5),80,60,510,170)+box(630,90,205,85,'discounted return',f(ret))+text(80,300,`contributions = ${contributions.map(v=>f(v)).join(' + ')}`,16),[['γ',p],['raw sum',9],['return',f(ret)]],'กราฟแท่งหาร contribution ด้วย 5 เพื่อพอดีกรอบ; ค่ารวม return แสดงในหน่วย reward จริง',`rewards=[0,1,3,5]\nG=sum(r*${p}**t for t,r in enumerate(rewards))  # ${f(ret)}`,{ret,contributions});
    }
    if(e==='noise') {
      const alpha=Math.exp(-p*0.45);let body=text(80,35,'x₀: CLEAN PATTERN',14,C.muted)+text(445,35,`x_t: t=${p}`,14,C.muted);
      for(let i=0;i<8;i++)for(let j=0;j<8;j++){const clean=(i===j||i+j===7)?1:-1,eps=Math.sin(i*73+j*31)*1.2,noisy=Math.sqrt(alpha)*clean+Math.sqrt(1-alpha)*eps,color=v=>`rgb(${Math.round(148+Math.max(-1,Math.min(1,v))*75)},${Math.round(166+Math.max(-1,Math.min(1,v))*65)},${Math.round(157+Math.max(-1,Math.min(1,v))*69)})`;body+=rect(80+j*29,55+i*29,27,27,color(clean),2)+rect(445+j*29,55+i*29,27,27,color(noisy),2)}
      return result(body,[['ᾱ_t',f(alpha)],['signal scale',f(Math.sqrt(alpha))],['noise scale',f(Math.sqrt(1-alpha))]],'noise pattern deterministic สำหรับการสาธิต; ในการฝึกจริง ε สุ่มจาก Gaussian ไม่ใช่ sine pattern นี้',`alpha_bar=exp(-0.45*${p})  # illustrative schedule\nx_t=sqrt(alpha_bar)*x0+sqrt(1-alpha_bar)*epsilon`,{alpha});
    }
    if(e==='similarity') {
      const scores=id===41?[0.8,0.2,-0.1]:[0.7,0.5,-0.2],prob=softmax(scores.map(v=>v/p));return result(bars(scores.map((v,i)=>id===41?`text ${i+1}`:`item ${i+1}`),prob,80,80,490,165)+box(630,75,205,80,'top candidate',1)+box(630,185,205,80,'top probability',f(prob[0])),[['τ',p],['scores',scores.join(', ')],['sum probabilities',f(sum(prob))]],'scores เป็นค่าตัวอย่างที่กำหนดไว้; temperature เปลี่ยน distribution แต่ไม่เปลี่ยนอันดับ',`scores=[${scores.join(', ')}]\np=softmax(scores/${p})  # ${prob.map(v=>f(v)).join(', ')}`,{prob});
    }
    if(e==='audio') {
      const vals=Array.from({length:96},(_,i)=>Math.sin(i*0.5)*0.6+Math.sin(i*1.7)*0.25),fs=16000;const body=rect(70,57,p*7.5,215,C.pale)+line(70,165,805,165)+path(vals.map((v,i)=>[70+i*7.5,165-v*95]))+text(70,305,`frame: ${p} samples · ${(p/fs*1000).toFixed(1)} ms at 16 kHz`,16)+text(620,40,'synthetic waveform',13,C.muted);return result(body,[['frame ms',f(p/fs*1000)],['Δf (Hz)',fs/p],['sample rate','16 kHz']],'waveform สังเคราะห์; frame ขนาดเล็กนี้ใช้ดูภาพง่าย ไม่ใช่ค่าแนะนำสำหรับ STT production',`sample_rate=16000; N=${p}\nframe_ms=N/sample_rate*1000  # ${f(p/fs*1000)}\nfreq_spacing=sample_rate/N  # ${fs/p}`,{duration:p/fs,frequency:fs/p});
    }
    if(e==='boxes') {
      const a=[130,75,180,160],b=[130+p,75+p/2,180,160],ix=Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0])),iy=Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1])),intersection=ix*iy,union=2*180*160-intersection,iou=intersection/union;
      return result(rect(...a,'#e6effb',2,`stroke="${C.blue}" stroke-width="3"`)+rect(...b,'none',2,`stroke="${C.orange}" stroke-width="3"`)+rect(Math.max(a[0],b[0]),Math.max(a[1],b[1]),ix,iy,'#a9cebe',0)+text(130,55,'blue: ground truth · orange: prediction',13,C.muted)+box(545,55,285,80,'IoU',f(iou))+box(545,160,285,80,'intersection / union',`${intersection} / ${union}`),[['shift x',p],['intersection',intersection],['IoU',f(iou)]],'ทั้งสองกล่องขนาด 180×160; shift y ครึ่งหนึ่งของ shift x',`intersection=${intersection}\nunion=2*180*160-intersection  # ${union}\niou=intersection/union       # ${f(iou)}`,{iou,intersection,union});
    }
    if(e==='retrieval') {
      const costs=[25,20,30,15],scores=[0.95,0.85,0.7,0.5];let used=0;const selected=costs.map(n=>{if(used+n<=p){used+=n;return true}return false});let body=costs.map((n,i)=>rect(65,45+i*68,430,53,selected[i]?C.pale:'#f1f1eb')+text(80,68+i*68,`chunk ${i+1} · relevance ${scores[i]}`,15,selected[i]?C.mint:C.muted)+text(80,86+i*68,`${n} tokens · ${selected[i]?'in context':'skipped: budget'}`,12,C.muted)).join('')+box(565,75,260,80,'tokens used / budget',`${used} / ${p}`)+box(565,180,260,80,'selected chunks',selected.filter(Boolean).length);
      return result(body,[['budget',p],['used tokens',used],['selected',selected.filter(Boolean).length]],'greedy scan ตามอันดับ: chunk ที่ไม่พอดีถูกข้าม แล้วลอง chunk ถัดไป; ไม่ได้ embed/retrieve หรือ generate จริง',`budget=${p}; used=0\nfor chunk in ranked_chunks:\n    if used+chunk.tokens <= budget:\n        context.append(chunk)\n        used += chunk.tokens`,{used,selected});
    }
    if(e==='agent') {
      const nodes=id===68?['input','context ACL','tool schema','authorize','execute','audit']:['request','model','tool call','validate','observe','answer'],done=Math.min(p,6);let body=nodes.map((n,i)=>{const x=55+(i%3)*275,y=55+Math.floor(i/3)*150;return box(x,y,245,90,`STEP ${i+1}`,n,i<done?C.mint:C.muted)+(i<5?text(x+248,y+55,i===2?'↓':'→',23,C.muted):'')}).join('');
      return result(body,[['budget',p],['stages reached',done],['status',done>=6?'complete':'budget exhausted']],id===68?'ลำดับ gates เป็น schematic; ผ่าน gate ในภาพไม่ได้ยืนยันความปลอดภัยของระบบจริง':'วงจร tool call จำลอง ไม่มีการเรียก model หรือ tool ภายนอก; แต่ละ stage ใช้งบหนึ่งหน่วยเพื่ออธิบาย bounded execution',`budget=${p}\nfor stage in pipeline:\n    if steps >= budget: stop('budget exhausted')\n    state = run_simulated_stage(state)\n    steps += 1`,{done});
    }
    if(e==='cache') {
      const L=4,B=1,T=id===48?64:p,H=id===48?p:4,D=16,b=2,bytes=2*L*B*T*H*D*b;let body='';for(let l=0;l<L;l++){body+=text(65,73+l*60,`layer ${l}`,14,C.muted)+rect(170,48+l*60,Math.min(460,T/256*460),37,C.pale)+rect(170,48+l*60,Math.min(460,T/256*460),17,C.mint,4)+text(660,72+l*60,`K+V · ${H} heads`,14)}body+=text(65,317,`B=${B} · T=${T} · Hkv=${H} · Dhead=${D} · FP16 (2 bytes)`,16);
      return result(body,[['tokens',T],['KV heads',H],['KV total KiB',f(bytes/1024)]],'memory คำนวณเฉพาะ K,V ของ 4 layers; ไม่รวม weights, allocator, temporary buffers และ cache metadata',`bytes=2*${L}*${B}*${T}*${H}*${D}*${b}\nkv_kib=bytes/1024  # ${f(bytes/1024)}`,{bytes,T,H});
    }
    if(e==='distributed') {
      const grads=Array.from({length:p},(_,i)=>1+i*0.2),mean=sum(grads)/p;let body=grads.map((g,i)=>{const x=50+i*790/p;return rect(x,60,790/p-9,80,C.pale)+text(x+(790/p-9)/2,89,`GPU ${i}`,12,C.muted,'middle')+text(x+(790/p-9)/2,121,f(g),21,C.mint,'middle')+line(x+(790/p-9)/2,150,445,209,C.line,2)}).join('')+box(315,220,260,80,'all-reduce mean',f(mean));return result(body,[['workers',p],['local batch',2],['global batch',p*2]],'local scalar gradients กำหนดไว้เพื่อสอน averaging; สมมติ workers มี local batch ขนาดเท่ากัน',`local_grads=[${grads.map(v=>f(v)).join(', ')}]\nglobal_grad=sum(local_grads)/${p}  # ${f(mean)}\nweights -= lr * global_grad`,{mean});
    }
    if(e==='gpu') {
      const N=32,blocks=Math.ceil(N/p);let body=text(65,35,`GRID: ${blocks} blocks × ${p} threads`,16);
      for(let i=0;i<N;i++){const x=65+(i%8)*94,y=65+Math.floor(i/8)*57,block=Math.floor(i/p);body+=rect(x,y,85,47,block%2?C.pale:'#edf1f8')+text(x+42,y+19,`b${block} t${i%p}`,11,C.muted,'middle')+text(x+42,y+38,`i=${i}`,15,block%2?C.mint:C.blue,'middle')}
      return result(body,[['elements',N],['block size',p],['blocks',blocks]],'thread index mapping จำลองบน CPU; ไม่ได้ benchmark หรือ launch GPU kernel จริง',`N=32; block_dim=${p}\nblocks=ceil(N/block_dim)  # ${blocks}\ni=block_idx*block_dim+thread_idx\nif i<N: c[i]=a[i]+b[i]`,{blocks});
    }
    if(e==='precision') {
      const g=1e-8,scaled=p*g,min=2**-24,quantized=Math.round(scaled/min)*min,unscaled=quantized/p;
      return result(box(55,78,225,90,'true gradient',g.toExponential())+box(335,78,225,90,'scaled gradient',scaled.toExponential(2))+box(620,78,225,90,'after FP16 round / S',unscaled.toExponential(2))+text(65,245,`FP16 minimum subnormal = 2⁻²⁴ ≈ ${min.toExponential(2)}`,17)+text(65,290,scaled<min/2?'ต่ำกว่าครึ่งหนึ่งของ grid: round เป็น 0':'ขยายจน represent เป็นค่าที่ไม่ใช่ 0 ได้',17,scaled<min/2?C.red:C.mint),[['scale S',p],['scaled FP16',quantized.toExponential(2)],['relative error',f(Math.abs(unscaled-g)/g)]],'toy FP16 rounding เฉพาะบริเวณ subnormal; อุปกรณ์/operations จริงอาจ flush subnormals เป็นศูนย์',`g=1e-8; S=${p}\nscaled=S*g\nfp16_rounded=round(scaled/2**-24)*2**-24\nunscaled=fp16_rounded/S`,{unscaled,quantized});
    }
    if(e==='lora') {
      let body=text(65,35,'FROZEN W · 8×8',14,C.muted)+rect(65,65,210,210,'#edf1f8')+text(340,35,`B · 8×${p}`,14,C.muted)+rect(340,65,p*24,210,C.pale)+text(610,35,`A · ${p}×8`,14,C.muted)+rect(610,65,210,p*24,C.pale)+text(290,181,'+',29)+text(550,181,'×',29)+text(65,315,`ΔW = BA · rank ≤ ${p}; α/r scaling แยกจาก parameter count`,16);
      return result(body,[['base params',64],['adapter params',16*p],['adapter/base',`${f(16*p/64*100)}%`]],'matrix 8×8 เล็กเพื่อให้เห็นขนาด; rank สูงอาจทำ adapter มี parameters มากกว่า base ในตัวอย่างนี้',`d_in=d_out=8; r=${p}\nA=zeros((r,8)); B=zeros((8,r))\nadapter_params=r*(8+8)  # ${16*p}\nW_eff=W+(alpha/r)*(B@A)`,{params:16*p});
    }
    if(e==='preference') {
      if(id===58){const reward=3,kl=0.8,shaped=reward-p*kl;return result(box(55,80,225,90,'reward model score',reward)+box(335,80,225,90,'β × KL penalty',f(p*kl),C.orange)+box(620,80,225,90,'shaped reward',f(shaped))+text(65,270,'reference คุมการเบี่ยง · reward model ยังอาจมี bias',17,C.muted),[['β',p],['KL',kl],['shaped reward',f(shaped)]],'reward=3 และ KL=0.8 เป็นค่าที่สมมติเพื่อแสดง penalty ไม่ใช่ผล RLHF training',`reward=3; KL=0.8; beta=${p}\nshaped_reward=reward-beta*KL  # ${f(shaped)}`,{shaped});}
      const margin=1.2-0.4,z=p*margin,prob=sigmoid(z),loss=-Math.log(prob);return result(box(55,80,225,90,'policy−reference margin',f(margin))+box(335,80,225,90,'DPO logit β × margin',f(z))+box(620,80,225,90,'DPO loss',f(loss),C.orange)+text(65,270,`σ(logit) = ${f(prob)} · คะแนนของคู่ตัวอย่าง ไม่ใช่ accuracy ทั้ง dataset`,17,C.muted),[['β',p],['logit',f(z)],['loss',f(loss)]],'สมมติ policy preference margin=1.2, reference=0.4; เปลี่ยน β โดยยังไม่ optimize policy',`policy_margin=1.2; reference_margin=0.4\nz=${p}*(policy_margin-reference_margin)\nloss=-log(sigmoid(z))  # ${f(loss)}`,{loss,z});
    }
    if(e==='evaluation') {
      const phat=0.8,z=1.96,den=1+z*z/p,center=(phat+z*z/(2*p))/den,half=z*Math.sqrt(phat*(1-phat)/p+z*z/(4*p*p))/den,lo=center-half,hi=center+half,sx=v=>70+v*740;
      return result(line(70,180,810,180)+line(sx(lo),145,sx(hi),145,C.mint,10)+line(sx(lo),125,sx(lo),166,C.mint,3)+line(sx(hi),125,sx(hi),166,C.mint,3)+circle(sx(phat),145,9,C.orange)+text(sx(lo),110,f(lo),15,C.mint,'middle')+text(sx(hi),110,f(hi),15,C.mint,'middle')+text(70,210,'0',13,C.muted)+text(810,210,'1',13,C.muted)+text(450,290,`Wilson 95% interval · observed ${Math.round(p*phat)}/${p} correct`,18,C.ink,'middle'),[['n',p],['accuracy',phat],['95% interval',`${f(lo)}–${f(hi)}`]],'สมมติ IID Bernoulli trials และ observed accuracy 0.8; interval ของ metric นี้ไม่ครอบคลุม bias/contamination ของ protocol',`n=${p}; p_hat=0.8; z=1.96\ncenter=(p_hat+z*z/(2*n))/(1+z*z/n)\nhalf=z*sqrt(p_hat*(1-p_hat)/n+z*z/(4*n*n))/(1+z*z/n)\ninterval=[center-half, center+half]`,{lo,hi});
    }
    if(e==='quant') {
      const qmax=2**(p-1)-1,s=1/qmax,w=0.37,q=Math.round(w/s),wHat=s*q,err=Math.abs(w-wHat);let body=line(70,155,815,155);
      const stride=Math.max(1,Math.ceil((2*qmax+1)/33));for(let i=-qmax;i<=qmax;i+=stride)body+=line(70+(i/qmax+1)*372.5,145,70+(i/qmax+1)*372.5,170,C.line,2);
      const wx=70+(w+1)*372.5,hx=70+(wHat+1)*372.5;body+=circle(wx,105,8,C.orange)+line(wx,110,wx,155,C.orange,2)+text(wx,78,`w=${w}`,17,C.orange,'middle')+circle(hx,195,8,C.mint)+line(hx,155,hx,190,C.mint,2)+text(hx,225,`ŵ=${f(wHat)}`,17,C.mint,'middle')+text(70,195,'−1',13,C.muted)+text(815,195,'+1',13,C.muted)+text(70,300,`scale=${f(s)} · q=${q} · absolute error=${f(err,5)}`,17);
      return result(body,[['bits',p],['scale',f(s)],['abs error',f(err,5)]],'symmetric signed quantization ใช้ q_min=−q_max; ไม่ใช้ค่าลบสุดที่ไม่สมมาตร; แสดง tick บางส่วนเมื่อ levels มาก',`q_max=2**(${p}-1)-1\nscale=1/q_max\nq=clip(round(0.37/scale),-q_max,q_max)\nw_hat=q*scale  # ${f(wHat)}`,{wHat,err});
    }
    if(e==='queue') {
      const mu=10,rho=p/mu,stable=p<mu,W=stable?1/(mu-p):Infinity,L=stable?p*W:Infinity;let body=rect(55,65,490,205,'#eef3f0')+text(75,93,'QUEUE (schematic)',13,C.muted);const boxes=Math.min(20,Math.ceil(stable?L:20));for(let i=0;i<boxes;i++)body+=rect(80+(i%10)*43,125+Math.floor(i/10)*60,34,43,stable?C.mint:C.red,5);body+=box(600,65,225,80,'utilization ρ',`${f(rho*100)}%`,stable?C.mint:C.red)+box(600,170,225,80,'mean time W (s)',stable?f(W):'unstable',stable?C.mint:C.red)+text(55,315,'M/M/1 toy model · μ=10/s · Poisson arrivals / exponential service',13,C.muted);
      return result(body,[['λ (/s)',p],['μ (/s)',mu],['W (s)',stable?f(W):'no steady-state']],'เป็น queue model สมมติ ไม่ใช่ serving benchmark; กล่องคิวเป็น schematic และกรณี λ≥μ ไม่มีค่ารอ steady-state ที่ finite',`lambda_rate=${p}; mu=10\nif lambda_rate < mu:\n    W=1/(mu-lambda_rate)  # ${stable?f(W):'undefined'}\nelse: overloaded=True`,{W,L,stable});
    }
    if(e==='release'||e==='capstone') {
      const labels=e==='capstone'?['corpus','tokenizer','random init','pretrain','evaluate','SFT / DPO','quantize','serve + monitor']:['artifact','start','ready','canary','promote'],columns=4;
      let body=labels.map((n,i)=>{const x=55+(i%columns)*200,y=60+Math.floor(i/columns)*145;return rect(x,y,182,100,i<=p?C.pale:'#f0f1ec')+text(x+15,y+25,`GATE ${i+1}`,12,C.muted)+text(x+15,y+56,n,19,i<=p?C.mint:C.muted)+text(x+15,y+82,i<p?'✓ artifact ready':i===p?'● current':'○ pending',12,C.muted)}).join('');
      return result(body,[['current',labels[p]],['prior gates',p],['total',labels.length]],'state machine สาธิตการส่ง artifact ต่อระหว่างขั้น; ไม่ได้ train หรือ deploy จริง',`stages=${JSON.stringify(labels)}\ncurrent=stages[${p}]\n# ต้องตรวจ artifact ของแต่ละ stage ก่อน advance`,{current:labels[p]});
    }
    if(e==='monitor') {
      const ref=[],now=[];for(let x=-3;x<=6;x+=0.1){ref.push([80+(x+3)*78,278-185*Math.exp(-x*x/2)]);now.push([80+(x+3)*78,278-185*Math.exp(-((x-p)**2)/2)])}
      return result(line(70,278,800,278)+path(ref,C.blue)+path(now,C.mint)+text(80,40,'REFERENCE N(0,1)',14,C.blue)+text(520,40,`NEW N(${p},1)`,14,C.mint)+text(80,315,'การเปลี่ยน P(x) เป็นสัญญาณตรวจ ไม่ได้บอก P(y|x) โดยตรง',16,C.muted),[['mean shift',p],['reference σ',1],['new σ',1]],'curves เป็น Gaussian densities ที่กำหนดไว้; ไม่ได้คำนวณ PSI หรือประเมิน model quality',`reference = Normal(mean=0, std=1)\nnew_data = Normal(mean=${p}, std=1)\n# investigate drift with actual quality labels`,{shift:p});
    }
    if(e==='attribution') {
      const cs=[2*p,-1],pred=0.5+sum(cs);return result(box(65,65,220,85,'baseline f(0,0)',0.5)+box(340,65,220,85,'x₁ contribution',f(cs[0]))+box(615,65,220,85,'x₂ contribution',-1,C.orange)+box(280,218,350,85,'prediction = baseline + contributions',f(pred)),[['x₁',p],['x₂',1],['prediction',f(pred)]],'linear contributions กับ baseline (0,0) คำนวณตรงตาม model นี้; ไม่ใช่ causal explanation',`f=lambda x1,x2: 2*x1-x2+0.5\ncontributions=[2*${p}, -1*1]\npred=0.5+sum(contributions)  # ${f(pred)}`,{pred,contributions:cs});
    }
    if(e==='compression') {
      const n=64,zero=Math.floor(n*p/100);let body='';for(let i=0;i<n;i++)body+=rect(80+(i%8)*29,50+Math.floor(i/8)*29,25,25,i<zero?'#e7ece8':C.mint,3);body+=box(440,55,365,80,'zero entries / total',`${zero} / 64`)+box(440,160,365,80,'dense FP32 bytes (unchanged)',256)+text(80,320,'สีเทาเป็น weights ที่ถูก mask → sparse kernel/representation ยังจำเป็น',15,C.muted);
      return result(body,[['requested sparsity',`${p}%`],['zeros',zero],['dense bytes',256]],'สมมติเรียง weights ตาม magnitude ก่อน mask; storage ยังคง dense 8×8 FP32',`mask = magnitude_rank >= ${zero}\nW_pruned = W * mask\ndense_bytes=8*8*4  # 256 even with zeros`,{zero,bytes:256});
    }
    if(e==='experts') {
      const logits=[2,1,0,-1],weights=softmax(logits.slice(0,p)),outputs=[1,2,3,4],output=sum(weights.map((w,i)=>w*outputs[i]));let body=box(65,55,180,85,'router logits','2,1,0,−1');for(let i=0;i<4;i++){const x=330+i*130;body+=line(245,100,x+53,155,i<p?C.mint:C.line,2)+rect(x,160,112,93,i<p?C.pale:'#f0f1ec')+text(x+56,190,`expert ${i+1}`,14,C.muted,'middle')+text(x+56,226,i<p?f(weights[i]):'off',21,i<p?C.mint:C.muted,'middle')}body+=text(330,307,`combined scalar output = ${f(output)}`,20,C.mint);
      return result(body,[['active',p],['total',4],['weighted output',f(output)]],'router logits และ expert outputs กำหนดไว้; weights normalize หลังเลือก top-k',`top_logits=[2,1,0,-1][:${p}]\nw=softmax(top_logits)\ny=sum(w * [1,2,3,4][:${p}])  # ${f(output)}`,{weights,output});
    }
    if(e==='reasoning') {
      const correct=[false,false,true,false,true,true].slice(0,p);let body=correct.map((v,i)=>rect(65+(i%3)*260,55+Math.floor(i/3)*130,240,100,v?C.pale:'#faece4')+text(80+(i%3)*260,85+Math.floor(i/3)*130,`candidate ${i+1}`,14,C.muted)+text(80+(i%3)*260,122+Math.floor(i/3)*130,v?'✓ verified':'× rejected',22,v?C.mint:C.orange)).join('');
      return result(body,[['candidates',p],['verified',correct.filter(Boolean).length],['cost units',p*10]],'candidate outcomes และ compute cost เป็นข้อมูลจำลองที่กำหนดไว้ เพื่อสอน search + verification ไม่ใช่ผล model benchmark',`candidates=generate(N=${p})\nvalid=[c for c in candidates if verifier(c)]\nanswer=select(valid) if valid else abstain()`,{verified:correct.filter(Boolean).length});
    }
    if(e==='world'||e==='robot') {
      let truth,model,note,code;if(e==='world'){truth=Array.from({length:p+1},(_,i)=>i*0.5);model=Array.from({length:p+1},(_,i)=>i*0.6);note='action=0.5 ต่อก้าว; model มี bias 0.1 ทำให้ error สะสมเป็น 0.1×horizon';code=`true_x=0; model_x=0\nfor t in range(${p}):\n    true_x += 0.5\n    model_x += 0.5 + 0.1`}
      else{truth=Array(9).fill(1);model=[0];for(let i=0;i<8;i++)model.push(model.at(-1)+p*(1-model.at(-1)));note='toy discrete unit-step plant ไม่มี delay/noise/saturation; stability ของ robot จริงต้องใช้ dynamics จริง';code=`x=0; target=1\nfor t in range(8):\n    error=target-x\n    x += ${p}*error`}
      const max=Math.max(1.5,...truth,...model),coords=a=>a.map((v,i)=>[80+i*710/(a.length-1),270-v/max*190]);let body=line(70,270,810,270)+path(coords(truth),C.orange,2,'stroke-dasharray="6 4"')+path(coords(model),C.mint)+text(80,40,e==='world'?'TRUE (orange) vs LEARNED (green)':'TARGET (orange) vs POSITION (green)',14,C.muted)+model.map((v,i)=>circle(...coords(model)[i],4,C.mint)).join('');
      return result(body,[['steps',model.length-1],['last prediction / x',f(model.at(-1))],['last abs error',f(Math.abs(truth.at(-1)-model.at(-1)))]],note,code,{truth,model});
    }
    if(e==='edge') {
      const bytes=100000*p/8,limit=300000,ram=64000;return result(box(65,65,315,85,'weight storage (decimal KB)',f(bytes/1000))+box(65,182,315,85,'runtime RAM example (KB)',ram/1000,C.blue)+rect(450,88,350,44,C.line)+rect(450,88,Math.min(350,350*bytes/limit),44,bytes<=limit?C.mint:C.red)+text(450,65,'FLASH BUDGET 300 KB (illustrative)',13,C.muted)+text(450,176,bytes<=limit?'✓ weights fit flash':'× weights exceed flash',21,bytes<=limit?C.mint:C.red)+text(450,240,'operator support + latency ยังต้องตรวจ',16,C.muted),[['parameters','100k'],['bits',p],['storage KB',bytes/1000]],'ตัวเลข budget/activation RAM สมมติ; weight bytes เป็นขั้นต่ำ ไม่รวม scales, metadata และ alignment',`params=100000; bits=${p}\nweight_bytes=params*bits/8  # ${bytes}\nfits_flash=weight_bytes<=300000\n# RAM and operator support checked separately`,{bytes});
    }
    throw new Error(`Unknown simulation engine: ${e}`);
  }
  window.LabSim={simulate,esc,f};
})();
