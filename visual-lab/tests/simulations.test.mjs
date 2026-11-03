import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const context={window:{}};
for(const name of ['lessons','chapter-meta','simulations'])vm.runInNewContext(fs.readFileSync(path.join(root,`visual-lab/assets/${name}.js`),'utf8'),context);
const lessons=context.window.VISUAL_LESSONS,simulate=context.window.LabSim.simulate;
const lesson=n=>lessons.find(l=>l.id===n);
const run=(n,p,stage=3)=>simulate(lesson(n),p,stage).values;
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-9,`${actual} ≠ ${expected}`);
test('80 chapters: valid content, reachable assets and all slider states produce valid SVG',()=>{
  assert.equal(lessons.length,80);assert.equal(new Set(lessons.map(l=>l.id)).size,80);
  let cases=0;
  for(const l of lessons){
    const meta=context.window.CHAPTER_META[l.id],p=l.parameter;
    assert.equal(l.steps.length,4);l.steps.forEach(s=>{assert.ok(s.title&&s.detail,`chapter ${l.id} invalid step`)});
    assert.ok(l.quiz.question&&l.quiz.answer&&l.quiz.distractor&&l.quiz.explanation);
    assert.notEqual(l.quiz.answer,l.quiz.distractor);assert.ok(p.min<=p.value&&p.value<=p.max&&p.step>0);
    for(const target of [`${meta.directory}/animation.html`,`${meta.directory}/README.md`,`${meta.directory}/exercises/README.md`,meta.source])assert.ok(fs.existsSync(path.join(root,target)),target);
    const html=fs.readFileSync(path.join(root,meta.directory,'animation.html'),'utf8');
    assert.ok(html.includes(`data-chapter="${l.id}"`));assert.ok(!/__CHAPTER__|__ASSETS__|__TITLE__/.test(html));
    for(const v of [p.value,p.min,p.max,...Array.from({length:Math.floor((p.max-p.min)/p.step)+1},(_,i)=>+(p.min+i*p.step).toFixed(6))]){
      for(let stage=0;stage<4;stage++){
        const sim=simulate(l,v,stage);assert.equal(sim.metrics.length,3);assert.ok(sim.code&&sim.note);
        assert.ok(!/NaN|undefined|Infinity/.test(sim.body),`invalid SVG: chapter ${l.id}, p=${v}, stage=${stage}`);
        assert.ok(!/(width|height|r)="-\d/.test(sim.body),`negative SVG dimensions: chapter ${l.id}`);cases++;
      }
    }
  }
  console.log(`Checked ${cases} parameter/stage combinations across 80 chapters.`);
});
test('filter excludes threshold equality',()=>{assert.equal(run(1,4).total,14);assert.equal(run(1,8).total,0)});
test('Bayes posterior normalizes and responds to base rate',()=>{near(run(9,0.5).post[1],0.8);assert.ok(run(9,0.2).post[1]<run(9,0.8).post[1])});
test('backprop matches finite differences away from ReLU kink',()=>{const w=0.8,h=1e-5,grad=(run(19,w+h).loss-run(19,w-h).loss)/(2*h);near(run(19,w).gradient,grad);near(run(18,-1).gradient,0)});
test('KNN selects nearest and k-means objective does not rise',()=>{assert.equal(run(8,3).selected.length,3);for(let i=1;i<=6;i++)assert.ok(run(14,i).inertia<=run(14,i-1).inertia+1e-9)});
test('attention weights normalize; causal and window masks have exactly zero mass',()=>{for(const n of [29,34,35,36,72]){const l=lesson(n);for(const p of [l.parameter.min,l.parameter.value,l.parameter.max]){const v=run(n,p);near(v.a.reduce((a,b)=>a+b,0),1);v.allowed.forEach((ok,i)=>{if(!ok)assert.equal(v.a[i],0)})}}assert.equal(run(35,1).a[0],1);assert.equal(run(72,1).a[3],1)});
test('convolution has known vertical edge response',()=>{const v=run(24,0);assert.equal(v.out.length,3);assert.deepEqual(Array.from(v.out[0]),[3,3,0])});
test('BPE merges reduce symbols and preserve text',()=>{assert.equal(run(49,0).count,6);assert.equal(run(49,3).count,2)});
test('RAG never exceeds budget and LoRA count follows dimensions',()=>{for(let p=20;p<=100;p+=10)assert.ok(run(46,p).used<=p);assert.equal(run(56,2).params,32)});
test('KV memory scales with heads and sequence length',()=>{near(run(48,8).bytes/run(48,2).bytes,4);near(run(62,128).bytes/run(62,64).bytes,2)});
test('IoU identity, finite error, and queue stability boundary',()=>{assert.equal(run(44,0).iou,1);assert.ok(run(44,100).iou<run(44,25).iou);assert.ok(run(61,8).err<run(61,2).err);assert.equal(run(63,9).stable,true);assert.equal(run(63,10).stable,false)});
test('Wilson intervals narrow with data and world bias compounds',()=>{const a=run(60,20),b=run(60,200);assert.ok(b.hi-b.lo<a.hi-a.lo);near(run(76,8).model.at(-1)-run(76,8).truth.at(-1),0.8)});
test('pruning does not change dense bytes; robot gains show overshoot',()=>{assert.equal(run(70,90).bytes,run(70,0).bytes);assert.ok(Math.max(...run(77,1.5).model)>1);assert.ok(Math.max(...run(77,0.5).model)<1)});
