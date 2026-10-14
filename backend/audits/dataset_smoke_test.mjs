import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = path.join(root, 'backend/audits/dataset-smoke-2026-09-21');
// HAM metadata has simple unquoted fields; reject other formats rather than misparse.
async function csv(relative) {
  const text = (await fs.readFile(path.join(root, relative), 'utf8')).trim();
  if (text.includes('"')) throw new Error('Quoted CSV requires a full CSV parser');
  const [header, ...lines] = text.split(/\r?\n/);
  const keys = header.split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((v,i) => [keys[i],v])));
}
const test = await csv('data/splits/test.csv');
const reference = await csv('results/train_embedding_metadata.csv');
const refLesions = new Set(reference.map(r => r.lesion_id));
const refImages = new Set(reference.map(r => r.image_id));
if (test.some(r => refLesions.has(r.lesion_id) || refImages.has(r.image_id))) throw new Error('Test/reference leakage');
const classes = ['nv','mel','bcc','bkl','akiec','vasc','df'];
const hash = r => crypto.createHash('sha256').update(`dataset-audit-42:${r.image_id}`).digest('hex');
const chosen = [];
for (const dx of classes) {
  const seen = new Set();
  const pool = test.filter(r => r.dx === dx).sort((a,b) => hash(a).localeCompare(hash(b)));
  for (const row of pool) {
    if (seen.has(row.lesion_id)) continue;
    chosen.push(row); seen.add(row.lesion_id);
    if (seen.size === 10) break;
  }
  if (seen.size !== 10) throw new Error(`Not enough independent lesions for ${dx}`);
}
await fs.mkdir(out, {recursive:true});
await fs.writeFile(path.join(out,'manifest.json'), JSON.stringify({ protocol:'10 distinct test lesions per class; deterministic SHA256 ordering with seed dataset-audit-42; zero image/lesion overlap with reference metadata; same images in both modes; top_k=10; default rarity gate remains enabled. No diagnosis sent to API. No ranking changes.', chosen },null,2));
const records = [];
for (const row of chosen) {
  let bytes;
  for (const part of [1,2]) {
    try { bytes = await fs.readFile(path.join(root,`data/HAM10000_images_part_${part}/${row.image_id}.jpg`)); break; }
    catch(e) { if (e.code !== 'ENOENT') throw e; }
  }
  if (!bytes) throw new Error(`Image missing: ${row.image_id}`);
  for (const mode of ['image_only','with_metadata']) {
    const form = new FormData();
    form.append('image', new Blob([bytes], {type:'image/jpeg'}), `${row.image_id}.jpg`);
    form.append('top_k','10');
    if (mode === 'with_metadata') {
      for (const key of ['age','sex','localization']) if (row[key] && row[key] !== 'unknown') form.append(key,row[key]);
    }
    try {
      const response = await fetch('http://127.0.0.1:8000/api/analyze', {method:'POST',body:form,signal:AbortSignal.timeout(90000)});
      const data = await response.json();
      await fs.writeFile(path.join(out,`${row.image_id}-${mode}.json`),JSON.stringify(data,null,2));
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${JSON.stringify(data)}`);
      const rs = data.results;
      const violations = [];
      if (rs.length !== 10) violations.push('count');
      for (const [i,r] of rs.entries()) {
        if (!refImages.has(r.image_id) || r.lesion_id === row.lesion_id) violations.push('reference/leakage');
        const s = r.scores;
        if (Math.abs(s.final_score-s.visual_similarity-s.metadata_score-s.rarity_adjustment)>0.000003) violations.push('score_sum');
        if (i && s.final_score>rs[i-1].scores.final_score) violations.push('ordering');
        if(s.visual_similarity<0.8 && s.rarity_adjustment!==0) violations.push('rarity_gate');
      }
      records.push({image_id:row.image_id,lesion_id:row.lesion_id,dx:row.dx,mode,top1:rs[0].diagnosis,top10:rs.map(r=>r.diagnosis),p5:rs.slice(0,5).filter(r=>r.diagnosis===row.dx).length/5,p10:rs.filter(r=>r.diagnosis===row.dx).length/10,hit10:rs.some(r=>r.diagnosis===row.dx),firstRelevantRank:rs.find(r=>r.diagnosis===row.dx)?.rank ?? null,rarityBoosted:rs.filter(r=>r.scores.rarity_adjustment>0).length,serverMs:data.processing_time_ms,violations});
    } catch(e) { records.push({image_id:row.image_id,dx:row.dx,mode,error:e.message}); }
  }
  if(records.length%20===0) console.log(`Completed ${records.length}/140 requests`);
  await fs.writeFile(path.join(out,'per-query.json'), JSON.stringify(records,null,2));
}
const summary=[];
for(const mode of ['image_only','with_metadata']) for(const dx of [...classes,'ALL']) {
  const rows=records.filter(r=>r.mode===mode&&(dx==='ALL'||r.dx===dx));
  const ok=rows.filter(r=>!r.error);
  const mean=key=>ok.reduce((s,r)=>s+Number(r[key]),0)/ok.length;
  summary.push({mode,dx,n:ok.length,errors:rows.length-ok.length,top1:ok.filter(r=>r.top1===r.dx).length,hit10:ok.filter(r=>r.hit10).length,p5:mean('p5'),p10:mean('p10'),meanServerMs:mean('serverMs'),queriesWithRarity:ok.filter(r=>r.rarityBoosted>0).length,violations:ok.reduce((s,r)=>s+r.violations.length,0)});
}
await fs.writeFile(path.join(out,'summary.json'),JSON.stringify(summary,null,2));
const fmt=x=>(x*100).toFixed(1)+'%';
let report='# Held-out HAM10000 live backend audit\n\n70 images: ten distinct test lesions per diagnosis, selected deterministically before inference. 140 requests on the same images, with and without available age/sex/location metadata. No diagnosis label was sent. Both modes retain default similarity-gated rarity reranking, so image_only is NOT a pure visual baseline. Entire test/reference metadata overlap check passed for image and lesion IDs.\n\n| Mode | Class | Queries | Rank-1 matches | Hit@10 | Precision@5 | Precision@10 |\n|---|---|---:|---:|---:|---:|---:|\n';
for(const s of summary) report+=`| ${s.mode} | ${s.dx} | ${s.n} | ${s.top1}/${s.n} | ${s.hit10}/${s.n} | ${fmt(s.p5)} | ${fmt(s.p10)} |\n`;
report+='\nThese are retrieval metrics on a balanced subset, not classification accuracy or estimates for the original imbalanced test distribution. No model/threshold changes or outcome-based image replacement. No NDCG is reported because the research implementation needs normalization correction. Query files preserve raw rankings, score components, and preview URLs.\n';
report+=`\nAPI errors: ${records.filter(r=>r.error).length}. Score/order/reference/gate violations: ${summary.filter(r=>r.dx==='ALL').reduce((s,r)=>s+r.violations,0)}.\n`;
await fs.writeFile(path.join(out,'REPORT.md'),report);
console.log(JSON.stringify(summary,null,2));
