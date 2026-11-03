/* Pure numerical models. A snapshot is the only input to the renderer. */
(function(root){
  'use strict';
  const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length;
  const dot=(a,b)=>{if(a.length!==b.length)throw Error(`shape mismatch: ${a.length} != ${b.length}`);return sum(a.map((v,i)=>v*b[i]));};
  const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
  const softmax=a=>{const m=Math.max(...a);if(!Number.isFinite(m))throw Error('fully masked row: ต้องมี key ที่อ่านได้อย่างน้อยหนึ่งตำแหน่ง');const e=a.map(v=>Math.exp(v-m)),z=sum(e);return e.map(v=>v/z);};
  const fmt=(v,d=5)=>typeof v==='number'?(Number.isFinite(v)?(v!==0&&(Math.abs(v)<1e-4||Math.abs(v)>=1e7)?v.toExponential(d):Number(v.toFixed(d)).toString()):String(v)):String(v);
  const clone=x=>JSON.parse(JSON.stringify(x));
  const numericClose=(actual,expected)=>Number.isFinite(actual)&&Number.isFinite(expected)&&Math.abs(actual-expected)<=Math.max(1e-12,Math.abs(expected)*1e-5);
  function freeze(x){if(x&&typeof x==='object'){Object.values(x).forEach(freeze);Object.freeze(x)}return x;}
  function snapshot(kind,metrics,nodes,table={},chart={},extra={}){return {kind,metrics,nodes,table,chart,tensors:[],warnings:[],note:'คำนวณจาก input ปัจจุบันใน browser; ไม่มีการเรียก model ภายนอก',...extra};}
  const node=(name,formula,value,explanation)=>({name,formula,value,explanation});
  const range=(key,label,min,max,step,value)=>({key,label,type:'range',min,max,step,value});
  const select=(key,label,options,value)=>({key,label,type:'select',options,value});
  const text=(key,label,value)=>({key,label,type:'text',value});
  const models={};
  function define(id,controls,compute,step){
    const checked=state=>{
      for(const c of controls){const value=state[c.key];
        if(c.type==='range'){
          if(typeof value!=='number'||!Number.isFinite(value))throw Error(`${c.label}: ต้องเป็นตัวเลข finite`);
          if(['lambda','eta','C','noise'].includes(c.key)&&value<0)throw Error(`${c.label}: ต้อง ≥ 0; ค่าติดลบเปลี่ยนความหมายของ objective/update`);
          if((['temperature','h','eps'].includes(c.key)||c.key==='epsilon'&&['dbscan','transformer'].includes(id))&&value<=0)throw Error(`${c.label}: ต้อง > 0 เพื่อไม่หารด้วยศูนย์`);
          if((['prior','like0','like1','confidence'].includes(c.key)||id==='gates'&&['f','i','o'].includes(c.key)||id==='reinforcement'&&['gamma','alpha'].includes(c.key))&&(value<0||value>1))throw Error(`${c.label}: ต้องอยู่ใน [0,1] ตามนิยาม probability/gate`);
          if(c.key==='beta'&&(value<0||id==='optimizer'&&value>=1||id==='preference'&&value===0))throw Error(`${c.label}: optimizer ต้อง 0≤β<1; penalty ต้องไม่ติดลบ และ preference β ต้อง >0`);
          if(c.key==='trainPercent'&&(value<=0||value>=100))throw Error('train percent ต้องอยู่ระหว่าง 0 และ 100 เพื่อให้มี train/holdout');
          if(c.key==='bits'&&(value<2||value>32||!Number.isInteger(value)))throw Error('bits ต้องเป็นจำนวนเต็ม 2–32 ใน lab นี้');
          if(c.step>=1&&(!Number.isInteger(value)||value<c.min||value>Math.max(4096,c.max)))throw Error(`${c.label}: ต้องเป็นจำนวนเต็ม ≥ ${c.min} และไม่เกิน ${Math.max(4096,c.max)} สำหรับ lab ขนาดเล็ก`);
          if((['column','queryPosition'].includes(c.key)||id==='language'&&c.key==='query')&&value>c.max)throw Error(`shape mismatch: ${c.label} เกิน index สูงสุด ${c.max}`);
        }else if(c.type==='select'&&!c.options.some(o=>o[0]===value))throw Error(`${c.label}: ตัวเลือกไม่ตรงกับที่กำหนด`);
      }
      const result=compute(state);
      const focusMap={convolution:[{tensors:['output Y']},{tableColumns:[0,1,2],tensors:['input X','selected patch']},{tableColumns:[2,3,4],tensors:['selected patch','kernel K']},{tableColumns:[4]},{tensors:['output Y']}],dbscan:[{tensors:['pairwise distances D']},{tableColumns:[0,3]},{tableColumns:[3,4]},{tableColumns:[5]},{tableColumns:[4,5]}],multihead:[{tableColumns:[1,2,3,4,5,6],tensors:['X','Q','K','V']},{tensors:['head0','head1']},{tensors:['head0 attention','head1 attention']},{tensors:['head0 attention','head1 attention']},{tableColumns:[7,8],tensors:['head0 output','head1 output','concatenated output']}],bootstrap:[{},{tableColumns:[0,1]},{tableColumns:[2]},{}],attention:[{tableColumns:[1,2,3],tensors:['K']},{tableColumns:[4]},{tableColumns:[5],tensors:['keep-mask']},{tableColumns:[6]},{tableColumns:[3,6,7]}],optimizer:[{tableColumns:[0,1,2]},{tableColumns:[2,3,4]},{tableColumns:[5,6,7]},{}]};
      if(focusMap[id])result.nodes.forEach((op,i)=>op.focus=focusMap[id][i]||{});

      const finite=o=>{if(typeof o==='number'&&!Number.isFinite(o))throw Error('คำนวณไม่ได้: พบ nonfinite value; ตรวจ input, divisor, scale และ learning rate');if(o&&typeof o==='object')Object.values(o).forEach(finite)};
      finite(result);return result;
    };
    models[id]={id,controls,defaults:Object.fromEntries(controls.map(c=>[c.key,c.value])),compute:checked,step};
  }
  define('regression',[
    range('w','weight w₁',-5,5,.05,1),range('b','bias b',-5,5,.05,0),range('eta','learning rate η',.001,1,.001,.1),
    range('sampleX','x ของจุดที่เลือก',-3,3,.1,2),range('sampleY','y ของจุดที่เลือก',-8,8,.1,5),
    select('degree','polynomial degree',[[1,'1'],[2,'2'],[3,'3'],[4,'4']],1),
    range('w2','weight x²',-3,3,.05,0),range('w3','weight x³',-3,3,.05,0),range('w4','weight x⁴',-3,3,.05,0),
    select('regularizer','penalty',[['none','ไม่มี'],['ridge','Ridge L2'],['lasso','Lasso L1']],'none'),range('lambda','regularization λ',0,3,.05,0),
    range('noise','noise scale (fixed pattern)',0,2,.1,0)
  ],s=>{
    const xs=[-2,-1,0,1,+s.sampleX],noise=[.4,-.7,.1,.6,-.4],ys=[-3,-1,1,3,+s.sampleY].map((y,i)=>y+s.noise*noise[i]);
    const degree=+s.degree,weights=[+s.w,+s.w2,+s.w3,+s.w4].slice(0,degree),basis=x=>weights.map((_,j)=>x**(j+1));
    const predict=x=>dot(weights,basis(x))+(+s.b),pred=xs.map(predict),errors=pred.map((v,i)=>v-ys[i]),squared=errors.map(e=>e*e),mse=mean(squared);
    const penalty=s.regularizer==='ridge'?s.lambda*sum(weights.map(w=>w*w)):s.regularizer==='lasso'?s.lambda*sum(weights.map(Math.abs)):0;
    const grad=weights.map((w,j)=>2*mean(errors.map((e,i)=>e*xs[i]**(j+1)))+(s.regularizer==='ridge'?2*s.lambda*w:s.regularizer==='lasso'?s.lambda*Math.sign(w):0));
    const gradB=2*mean(errors),loss=mse+penalty,valX=[-1.5,.5,1.5],valY=valX.map(x=>2*x+1),validation=mean(valX.map((x,i)=>(predict(x)-valY[i])**2));
    const finite=[loss,gradB,...grad,...pred].every(Number.isFinite);if(!finite)throw Error('diverged: พบค่าที่ไม่ finite; ย้อนหนึ่งก้าวและลด learning rate');
    const curve=Array.from({length:81},(_,i)=>{const x=-3+i*.075;return [x,predict(x)]});
    const nodes=[node('ออกแบบ features',`φ(x)=[x, …, x^${degree}]`,`${xs.length} × ${degree}`,'แต่ละแถวเป็นตัวอย่าง แต่ละคอลัมน์เป็นกำลังของ x; polynomial ยัง linear in weights'),
      node('คูณแล้วบวก bias',`ŷᵢ=Σⱼwⱼxᵢ^(j+1)+b`,pred,'w เปลี่ยน slope/curvature; bias เลื่อนทุก prediction เท่ากัน'),
      node('วัด residual',`eᵢ=ŷᵢ−yᵢ`,errors,'นิยาม residual ใช้ prediction ลบ target ตลอดทั้งตัวอย่าง'),
      node('รวม objective',`MSE=Σeᵢ²/${xs.length}; J=MSE+penalty`,loss,`MSE=${fmt(mse)}, penalty=${fmt(penalty)}; ไม่ลงโทษ bias`),
      node('chain rule ถึง weights','∂J/∂wⱼ = mean(2e x^(j+1)) + penalty derivative',grad,'คูณ local derivative แล้วเฉลี่ยตัวอย่าง; gradient ติดลบทำให้ weight เพิ่มเมื่อ update'),
      node('chain rule ถึง bias','∂J/∂b=mean(2e)',gradB,'เพราะ ∂(wx+b−y)/∂b=1 ทุกตัวอย่าง'),
      node('เสนอ optimizer step',`w_new=w−${s.eta}g; b_new=b−${s.eta}g_b`,[...weights.map((w,j)=>w-s.eta*grad[j]),s.b-s.eta*gradB],'นี่เป็นข้อเสนอก้าวถัดไป กด update จึงเปลี่ยนสถานะจริง')];
    const focusColumns=[[0],[0,1,2],[2,1,3],[3,4],[3,0,5],[3],[]];nodes.forEach((op,i)=>op.focus={tableColumns:focusColumns[i]});
    return snapshot('regression',[['MSE',mse],['objective J',loss],['∂J/∂w₁',grad[0]],['∂J/∂b',gradB],['validation MSE',validation]],nodes,
      {columns:['x','target y','prediction ŷ','e=ŷ−y','e²','2ex'],rows:xs.map((x,i)=>[x,ys[i],pred[i],errors[i],squared[i],2*errors[i]*x])},
      {type:'xy',xLabel:'x',yLabel:'y / prediction',series:[{label:'training observations',points:xs.map((x,i)=>[x,ys[i]]),dots:true},{label:'prediction ปัจจุบัน',points:curve}],residuals:xs.map((x,i)=>[x,ys[i],pred[i]])},
      {loss,mse,penalty,gradient:grad,gradB,predictions:pred,errors,xs,ys,weights,validation,
       warnings:Math.abs(s.w)>1e5||loss>1e12?['loss โตมาก: ตรวจ scale และลด η; ค่าจริงยังแสดงโดยไม่ปิดบัง']:[],
       surfacePoint:{w:s.w,b:s.b},surfaceContext:JSON.stringify({xs,ys,degree,otherWeights:weights.slice(1),regularizer:s.regularizer,lambda:s.lambda}),
       surface:Array.from({length:17},(_,i)=>Array.from({length:17},(_,j)=>{const w=-1+j*.375,b=-2+i*.375,ws=[w,...weights.slice(1)],p=s.regularizer==='ridge'?s.lambda*sum(ws.map(v=>v*v)):s.regularizer==='lasso'?s.lambda*sum(ws.map(Math.abs)):0;return {w,b,loss:mean(xs.map((x,k)=>(dot(ws,basis(x))+b-ys[k])**2))+p}}))});
  },(s,r)=>{const keys=['w','w2','w3','w4'],next={...s,b:s.b-s.eta*r.gradB};r.gradient.forEach((g,i)=>next[keys[i]]=s[keys[i]]-s.eta*g);return next;});
  class Session{
    constructor(model,state){this.model=typeof model==='string'?models[model]:model;this.entries=[];this.cursor=-1;this.baseline=null;this.commit({...this.model.defaults,...state},'initial');this.pin();}
    get current(){return this.entries[this.cursor]}
    commit(state,event='manual'){const result=this.model.compute(clone(state));const entry=freeze({state:clone(state),result,event,index:this.cursor+1});this.entries=this.entries.slice(0,this.cursor+1);this.entries.push(entry);this.cursor++;return entry;}
    edit(key,value){let state={...this.current.state,[key]:value};if(this.model.id==='optimizer'&&key==='method')state={...state,velocity:0,firstMoment:0,secondMoment:0,updates:0};if(this.model.id==='optimizer'&&key==='initialTheta')state={...state,theta:value,velocity:0,firstMoment:0,secondMoment:0,updates:0};return this.commit(state,`manual:${key}`)}
    update(){if(!this.model.step)throw Error('กลไกนี้ไม่มี optimizer step ให้ใช้ controls หรือตัวเดิน operation');return this.commit(this.model.step(clone(this.current.state),this.current.result),'algorithm:update')}
    back(){if(this.cursor>0)this.cursor--;return this.current}
    forward(){if(this.cursor<this.entries.length-1)this.cursor++;return this.current}
    pin(){this.baseline=freeze(clone(this.current));return this.baseline}
    reset(){this.entries=[];this.cursor=-1;this.commit({...this.model.defaults},'reset');this.pin();return this.current}
  }
  root.CourseMath={models,define,range,select,text,snapshot,node,sum,mean,dot,sigmoid,softmax,fmt,clone,numericClose,Session};
})(typeof window==='undefined'?globalThis:window);
