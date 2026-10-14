import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const out = path.join(path.dirname(fileURLToPath(import.meta.url)), 'web-smoke-2026-09-21');
const cases = [
  { dx: 'nv', source: 'https://dermnetnz.org/topics/melanocytic-naevus', image: 'https://dermnetnz.org/assets/Uploads/lesions/reticular.jpg', label: 'Dermoscopy: reticular naevus' },
  { dx: 'mel', source: 'https://dermnetnz.org/cme/dermoscopy-course/dermoscopy-of-melanoma', image: 'https://dermnetnz.org/assets/Uploads/doctors/dermoscopy-course/images/mel-chg.jpg', label: 'Melanoma; irregular structure' },
  { dx: 'bcc', source: 'https://dermnetnz.org/topics/basal-cell-carcinoma-dermoscopy', image: 'https://dermnetnz.org/assets/Uploads/2010-quizzes/2010-July-3-bcc-ovoid.jpg', label: 'Pigmented basal cell carcinoma dermoscopy' },
  { dx: 'bkl', source: 'https://commons.wikimedia.org/wiki/File:Dermatoscopy_SebK.jpg', image: 'https://upload.wikimedia.org/wikipedia/commons/d/d9/Dermatoscopy_SebK.jpg', label: 'Seborrheic keratosis, polarized dermoscopy; P. Tschandl, CC BY-SA 4.0' },
  { dx: 'akiec', source: 'https://dermnetnz.org/topics/actinic-keratosis-dermoscopy', image: 'https://dermnetnz.org/assets/Uploads/strawberry-pattern-ak-4.jpg', label: 'Strawberry pattern dermoscopy of facial actinic keratosis' },
  { dx: 'vasc', source: 'https://dermnetnz.org/imagedetail/6653-cherry-angioma-dermoscopy', image: 'https://dermnetnz.org/assets/Uploads/Cherry-angioma-dermoscopy6.jpg', label: 'Cherry angioma dermoscopy' },
  { dx: 'df', source: 'https://dermnetnz.org/imagedetail/8628-dermatofibroma', image: null, label: 'Dermoscopy of dermatofibroma' },
];

await fs.mkdir(out, { recursive: true });
const summary = [];
for (const c of cases) {
  try {
    if (!c.image) {
      const page = await fetch(c.source, { signal: AbortSignal.timeout(30000) });
      const html = await page.text();
      const tag = html.match(/<img\b[^>]*js-gallery-image[^>]*>/i)?.[0];
      const src = tag?.match(/\bsrc="([^"]+)"/)?.[1];
      if (!src) throw new Error('No gallery image found');
      c.image = new URL(src, c.source).href;
    }
    const response = await fetch(c.image, { signal: AbortSignal.timeout(30000) });
    if (!response.ok) throw new Error(`Image HTTP ${response.status}`);
    const type = response.headers.get('content-type') || '';
    if (!type.startsWith('image/')) throw new Error(`Not an image: ${type}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    await fs.writeFile(path.join(out, `${c.dx}.jpg`), bytes);
    c.sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
    const form = new FormData();
    form.append('image', new Blob([bytes], { type }), `${c.dx}.jpg`);
    form.append('top_k', '10');
    const start = Date.now();
    const api = await fetch('http://127.0.0.1:8000/api/analyze', { method: 'POST', body: form, signal: AbortSignal.timeout(120000) });
    const data = await api.json();
    await fs.writeFile(path.join(out, `${c.dx}-response.json`), JSON.stringify(data, null, 2));
    if (!api.ok) throw new Error(`API ${api.status}: ${JSON.stringify(data)}`);
    const results = data.results;
    const count = k => results.slice(0, k).filter(r => r.diagnosis === c.dx).length;
    const row = { expected: c.dx, top1: results[0]?.diagnosis, top10: results.map(r => r.diagnosis), matches5: count(5), matches10: count(10), firstRelevantRank: results.find(r => r.diagnosis === c.dx)?.rank ?? null, responseMs: Date.now()-start, serverMs: data.processing_time_ms, topSimilarity: results[0]?.scores.visual_similarity, rarityBoostedCount: results.filter(r => r.scores.rarity_adjustment > 0).length, metadataScoresZero: results.every(r => r.scores.metadata_score === 0), returnedCount: results.length };
    summary.push(row);
    console.log(JSON.stringify(row));
  } catch (e) { const row = { expected: c.dx, error: e.message }; summary.push(row); console.log(JSON.stringify(row)); }
  await fs.writeFile(path.join(out, 'manifest.json'), JSON.stringify({ date: new Date().toISOString(), protocol: 'One preselected labeled dermoscopic image per HAM class; unchanged downloads; no patient metadata; top_k=10; no outcome-based replacements. Source labels are not independently verified histology. Web origin does not establish independence from HAM10000. DermNet/licensor copyright retained; private audit only, no publication license granted.', cases }, null, 2));
  await fs.writeFile(path.join(out, 'summary.json'), JSON.stringify(summary, null, 2));
}
