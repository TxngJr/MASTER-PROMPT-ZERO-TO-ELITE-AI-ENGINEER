/* Criterion evidence, not a claim about a learner's mind or identity. */
(function(root){'use strict';
const copy=x=>JSON.parse(JSON.stringify(x)),isObject=x=>x&&typeof x==='object'&&!Array.isArray(x),text=x=>typeof x==='string'&&x.trim().length>0,hex=x=>typeof x==='string'&&/^[0-9a-f]{64}$/.test(x),fields=['explanation','calculation','implementation','limits'];
const snapshot=entry=>Object.fromEntries(fields.map(k=>[k,typeof entry?.[k]==='string'?entry[k]:'']));
function timestamp(value){if(typeof value!=='string')return NaN;const m=value.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?(Z|([+-])(\d{2}):(\d{2}))$/);if(!m)return NaN;const at=Date.parse(value);if(!Number.isFinite(at))return NaN;const offset=m[3]==='Z'?0:(m[4]==='+'?1:-1)*(+m[5]*60 + +m[6])*60000;return new Date(at+offset).toISOString().slice(0,19)===m[1]?at:NaN;}
function canonical(x){if(Array.isArray(x))return '['+x.map(canonical).join(',')+']';if(isObject(x))return '{'+Object.keys(x).sort().map(k=>JSON.stringify(k)+':'+canonical(x[k])).join(',')+'}';return JSON.stringify(x);}
function runIdentity(r){return {...Object.fromEntries(['sourceSha256','suiteHash','seed','testedAt','sourceRole','contentVersion','chapter','suiteFamily','python','exitCode','passed','failed','skipped'].map(k=>[k,r[k]])),checks:r.checks.map(c=>({id:c.id,status:c.status}))};}
async function digest(x){if(!root.crypto?.subtle)throw Error('เบราว์เซอร์นี้ไม่มี SHA-256 สำหรับตรวจความสอดคล้องของรายงาน ใช้ localhost บนเบราว์เซอร์ที่รองรับ');const result=await root.crypto.subtle.digest('SHA-256',new TextEncoder().encode(canonical(x)));return [...new Uint8Array(result)].map(v=>v.toString(16).padStart(2,'0')).join('');}
function common(packet,book){if(!isObject(packet)||packet.schemaVersion!==1||packet.chapter!==book.id||packet.contentVersion!==book.contentVersion)throw Error('รายงานไม่ตรงบทหรือเวอร์ชันเนื้อหาปัจจุบัน เก็บผลเก่าไว้เป็นประวัติแล้วประเมินรุ่นนี้ใหม่');}
async function validateRun(packet,book){
 common(packet,book);const expected=book.runtimeAssessment;
 if(packet.kind!=='learner-code-run'||packet.sourceRole!=='learner-submission')throw Error('ต้องเป็นผลรันไฟล์ของผู้เรียน รายงาน QA/ต้นแบบไม่ใช้รับรองผู้เรียน');
 if(!expected||packet.suiteHash!==expected.suiteHash||(packet.suiteFamily||'regression-challenge')!==expected.family)throw Error('ชุดตรวจไม่ตรงกับ suite ที่กำหนดในบทนี้ กรุณารันตัวตรวจรุ่นปัจจุบัน');
 if(!hex(packet.sourceSha256)||!hex(packet.reportId)||!Array.isArray(packet.checks)||packet.checks.length!==expected.checkCount)throw Error('รายงานขาด hash หรือรายการตรวจครบชุด');
 const ids=new Set();for(const c of packet.checks){if(!text(c.id)||ids.has(c.id)||!['pass','fail','skip'].includes(c.status))throw Error('check ID ซ้ำหรือสถานะไม่ถูกต้อง');ids.add(c.id);}
 if(expected.checkIds&&expected.checkIds.some(id=>!ids.has(id)))throw Error('รายงานขาด required check ของกลไกนี้');
 for(const[name,status]of[['passed','pass'],['failed','fail'],['skipped','skip']])if(packet[name]!==packet.checks.filter(c=>c.status===status).length)throw Error('จำนวนผลตรวจไม่ตรงกับรายการ checks');
 if(!Number.isInteger(packet.exitCode)||!Number.isInteger(packet.seed)||packet.seed<0||packet.seed>=2**32||!Number.isFinite(timestamp(packet.testedAt)))throw Error('metadata ผลรันไม่ครบ หรือวันที่ไม่ใช่ ISO 8601 ที่มีอยู่จริง');
 if(await digest(runIdentity(packet))!==packet.reportId)throw Error('ข้อมูลรายงานเปลี่ยนจาก checksum กรุณารันใหม่กับ source ที่ส่งจริง');
 return copy(packet);
}
function validateReview(packet,book){
 common(packet,book);const policy=book.masteryPolicy;
 if(packet.kind!=='criterion-review'||packet.policyVersion!==policy.version||!isObject(packet.reviewer)||!isObject(packet.learner)||!Array.isArray(packet.objectives))throw Error('รูปแบบรายงานผู้ตรวจไม่ตรง template ของบทนี้');
 if(packet.objectives.length!==book.objectiveContracts.length)throw Error('ใช้แบบฟอร์มครบทุก objective ของบทนี้ ข้อที่ยังไม่ได้ประเมินให้เก็บ score=null เพื่อไม่ทำผลข้ออื่นหาย');
 const allowed=new Set(book.objectiveContracts.map(o=>o.id)),seen=new Set();
 for(const entry of packet.objectives){if(!isObject(entry)||!allowed.has(entry.objectiveId)||seen.has(entry.objectiveId)||!Array.isArray(entry.rounds)||entry.rounds.length>3)throw Error('objective ID ไม่ตรง ซ้ำ หรือรอบประเมินเกิน template');seen.add(entry.objectiveId);
  if(!isObject(entry.submission)||fields.some(k=>typeof entry.submission[k]!=='string'))throw Error('ขาด snapshot งานที่ผู้ตรวจใช้ประเมิน');
  const rounds=new Set();for(const r of entry.rounds){if(!isObject(r)||!policy.rounds.some(p=>p.id===r.kind)||rounds.has(r.kind)||!isObject(r.rubric))throw Error('รอบประเมินไม่ถูกต้องหรือซ้ำ');rounds.add(r.kind);for(const d of policy.dimensions){const s=r.rubric[d.id];if(!isObject(s)||![null,0,1,2].includes(s.score)||typeof s.note!=='string'||typeof s.reference!=='string')throw Error('rubric ต้องระบุ score 0/1/2 หรือ null พร้อม note/reference');}}
 }
 return copy(packet);
}
function reviewTemplate(book,record){return {schemaVersion:1,kind:'criterion-review',chapter:book.id,contentVersion:book.contentVersion,policyVersion:book.masteryPolicy.version,learner:{name:''},reviewer:{name:'',role:''},notice:'ผู้ตรวจกรอกผลที่สังเกตจริง score2 ทุกด้านในทุกครั้ง ไม่มีการให้คะแนนล่วงหน้า การนำเข้าไม่ยืนยันตัวตนผู้ตรวจหรือความแท้ของเอกสาร',scoreAnchors:book.masteryPolicy.scoreAnchors,runtimeCandidates:Object.values(record.runtimeReports||{}).filter(r=>r.contentVersion===book.contentVersion).map(r=>({reportId:r.reportId,sourceSha256:r.sourceSha256,passed:r.passed,failed:r.failed,skipped:r.skipped})),objectives:book.objectiveContracts.map(o=>({objectiveId:o.id,task:book.evidence.find(e=>e.objectiveId===o.id)?.task,sections:o.sections,submission:snapshot(record.evidence?.[o.evidenceIds[0]]),runReportId:'',reviewerVerifiedSourceSha256:'',rounds:book.masteryPolicy.rounds.map(p=>({kind:p.id,caseId:'',caseFingerprint:'',caseDescription:'',at:'',assistance:'',rubric:Object.fromEntries(book.masteryPolicy.dimensions.map(d=>[d.id,{score:null,note:'',reference:'',criterion:d.pass}]))}))}))};}
function objectiveStatus(book,record,o,now=Date.now()){
 const pending=[],entry=record.evidence?.[o.evidenceIds[0]],review=record.criterionReview,currentReview=review?.contentVersion===book.contentVersion&&review?.policyVersion===book.masteryPolicy.version,submitted=currentReview?review.objectives?.find(e=>e.objectiveId===o.id):null;
 if(fields.some(k=>!text(entry?.[k]))||entry?.contentVersion!==book.contentVersion)pending.push('เขียนงานทั้ง 4 ส่วนสำหรับเนื้อหารุ่นนี้');
 if(!currentReview)pending.push('ยังไม่มีรายงานผู้ตรวจรุ่นนี้');
 else {
  if(!text(review.learner?.name)||!text(review.reviewer?.name)||!['instructor','peer'].includes(review.reviewer?.role)||review.learner.name.trim()===review.reviewer.name.trim())pending.push('ต้องมีรายงานจากผู้ตรวจคนอื่น ชื่อและบทบาทชัดเจน; self review ไม่นับ');
  if(!submitted)pending.push('ผู้ตรวจยังไม่ประเมิน objective นี้');
 }
 if(submitted){
  if(canonical(submitted.submission)!==canonical(snapshot(entry)))pending.push('ข้อความงานเปลี่ยนหลังตรวจ ส่งงานรุ่นปัจจุบันให้ตรวจใหม่');
  const run=record.runtimeReports?.[submitted.runReportId],expected=book.runtimeAssessment;
  if(!run||run.contentVersion!==book.contentVersion||run.suiteHash!==expected?.suiteHash||run.sourceRole!=='learner-submission'||run.exitCode!==0||run.failed!==0||run.skipped!==0||run.passed!==expected?.checkCount||submitted.reviewerVerifiedSourceSha256!==run.sourceSha256)pending.push('ขาดผลรัน source ผู้เรียนครบชุด หรือผู้ตรวจยังไม่ยืนยัน hash ไฟล์ตรงรายงาน');
  else if(!Number.isFinite(timestamp(run.testedAt))||timestamp(run.testedAt)>now)pending.push('วันที่รันไม่ถูกต้อง/อยู่ในอนาคต ตรวจเวลาแล้วรันใหม่');
  const cases=new Set(),fingerprints=new Set(),times={};
  for(const p of book.masteryPolicy.rounds){const r=submitted.rounds.find(r=>r.kind===p.id);if(!r){pending.push(p.label+': ยังไม่มีผล');continue}
   const at=timestamp(r.at);times[p.id]=at;
   if(!text(r.caseId)||!hex(r.caseFingerprint)||!text(r.caseDescription)||cases.has(r.caseId)||fingerprints.has(r.caseFingerprint))pending.push(p.label+': ต้องมีโจทย์ต่างกรณีและ fingerprint ไม่ซ้ำ');cases.add(r.caseId);fingerprints.add(r.caseFingerprint);
   if(!Number.isFinite(at)||at>now)pending.push(p.label+': วันที่ประเมินไม่ถูกต้อง/อยู่ในอนาคต');
   if(r.assistance!=='none')pending.push(p.label+': ยังไม่ได้ทำโดยไม่ใช้ hints/เฉลย');
   for(const d of book.masteryPolicy.dimensions){const s=r.rubric[d.id];if(s?.score!==2||!text(s.note)||!text(s.reference))pending.push(p.label+' · '+d.label+': ต้องได้ 2 พร้อมเหตุผลและหลักฐาน');}
  }
  if(Number.isFinite(times.initial)&&Number.isFinite(times.transfer)&&times.transfer<times.initial)pending.push('รอบ transfer ต้องไม่มาก่อนการประเมินครั้งแรก');
  if(Number.isFinite(times.initial)&&Number.isFinite(times.retention)&&(times.retention-times.initial<book.masteryPolicy.retentionDays*86400000||Number.isFinite(times.transfer)&&times.retention<times.transfer))pending.push('ยังไม่ครบช่วงทบทวนอย่างน้อย '+book.masteryPolicy.retentionDays+' วันหลังครั้งแรก');
 }
 return {objectiveId:o.id,status:pending.length?'insufficient-evidence':'reported-criteria-met',pending,section:o.sections[0]};
}
function chapterStatus(book,record={},now=Date.now()){
 const rows=book.objectiveContracts.map(o=>objectiveStatus(book,record,o,now)),covered=rows.filter(r=>r.status==='reported-criteria-met').length;
 const web=book.checks.every(q=>record.practice?.[q.id]?.version===book.contentVersion),novel=book.checks.filter(q=>q.variants.length>1).every(q=>record.retests?.[q.id]?.version===book.contentVersion);
 return {chapter:book.id,contentVersion:book.contentVersion,covered,total:rows.length,rows,webPracticeComplete:web,webNewCasesComplete:novel,criteriaReportedComplete:rows.length>0&&covered===rows.length&&web&&novel,guaranteesUnderstanding:false,reviewerIdentityVerified:false};
}
function courseStatus(books,records={},now=Date.now()){const chapters=Object.values(books).map(b=>chapterStatus(b,records[b.id]||{},now)),learners=[...new Set(Object.values(books).map(b=>records[b.id]?.criterionReview).filter(r=>r&&r.contentVersion===books[r.chapter]?.contentVersion&&text(r.learner?.name)).map(r=>r.learner.name.trim()))],learnerConflict=learners.length>1;return {coveredObjectives:learnerConflict?0:chapters.reduce((n,c)=>n+c.covered,0),totalObjectives:chapters.reduce((n,c)=>n+c.total,0),chaptersReportedComplete:learnerConflict?0:chapters.filter(c=>c.criteriaReportedComplete).length,totalChapters:chapters.length,learnerConflict,declaredLearners:learners,guaranteesUnderstanding:false,reviewerIdentityVerified:false,chapters};}
function storeReview(record,packet){record.reviewArchive=record.reviewArchive||[];if(record.criterionReview)record.reviewArchive.push(copy(record.criterionReview));record.criterionReview=copy(packet);}
function restoreReview(record){if(!record.reviewArchive?.length)return false;const previous=record.reviewArchive.pop();if(record.criterionReview)record.reviewArchive.push(copy(record.criterionReview));record.criterionReview=copy(previous);return true;}
root.CourseMastery={snapshot,canonical,runIdentity,digest,validateRun,validateReview,reviewTemplate,objectiveStatus,chapterStatus,courseStatus,storeReview,restoreReview};
})(typeof window==='undefined'?globalThis:window);
