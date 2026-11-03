#!/usr/bin/env node
// Generate deterministic, file:// compatible entry points for all chapters.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const lab=path.join(root,'visual-lab');
const ctx={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(lab,'assets/lessons.js'),'utf8'),ctx);
const lessons=ctx.window.VISUAL_LESSONS;
if(lessons.length!==80)throw new Error('Expected 80 lessons');
const meta={},phase=n=>n<=3?'A':n<=17?'B':n<=28?'C':n<=36?'D':n<=44?'E':n<=60?'F':n<=67?'G':n<=78?'H':'I';
for(const lesson of lessons){
  const directory=fs.readdirSync(root).find(d=>d.startsWith(`${String(lesson.id).padStart(2,'0')}-`)&&fs.statSync(path.join(root,d)).isDirectory());
  const md=fs.readFileSync(path.join(root,directory,'README.md'),'utf8');
  const english=md.split('\n')[0].replace(/^# Chapter \d+ — /,'');
  const topics=md.split('\n').filter(l=>/^## \d+\. /.test(l)).map(l=>l.replace(/^## \d+\. /,'')).filter(t=>!/(Why|Goal|Prerequisites|Learning Objectives|Mental Model|Study Order|Checklist|Next|Exercises|Installation|Common Mistakes)/i.test(t)).slice(0,7);
  const sourceFiles=fs.readdirSync(path.join(root,directory,'src')).filter(f=>f.endsWith('.py')&&f!=='__init__.py').sort();
  meta[lesson.id]={directory,english,phase:phase(lesson.id),topics,source:`${directory}/src/${sourceFiles[0]}`};
  if(lesson.steps.length!==4||!Number.isFinite(lesson.parameter.min)||!Number.isFinite(lesson.parameter.value)||!lesson.quiz.explanation)throw new Error(`Invalid lesson ${lesson.id}`);
}
fs.writeFileSync(path.join(lab,'assets/chapter-meta.js'),`// Generated from chapter README files by tools/build_visual_lab.mjs\nwindow.CHAPTER_META = ${JSON.stringify(meta,null,2)};\n`);
const python=process.env.VISUAL_LAB_PYTHON|| (fs.existsSync(path.join(root,'.venv/bin/python'))?path.join(root,'.venv/bin/python'):'python3');
const built=spawnSync(python,['tools/build_visual_book.py',...process.argv.slice(2)],{cwd:root,stdio:'inherit'});
if(built.error)throw built.error;
process.exitCode=built.status??1;
