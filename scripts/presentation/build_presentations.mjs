/**
 * 저장된 실측 JSON 여덟 개와 한글 원고로 A2_1 발표자료 두 벌을 만든다.
 * --modules, --skill-dir, --python에는 제공된 실행 환경의 경로를 전달한다.
 * 완성 PPTX/PDF는 덮어쓰지 않고, 초안·렌더·배치·검증 기록은 --build-dir에 둔다.
 * 최초 제작 표식은 상위 작업에서 한 번 실행하므로 이 파일에서는 호출하지 않는다.
 * --skip-pdf는 PPTX 검토 단계이며, --pdf-only는 검토된 PPTX의 해시를 확인한 뒤
 * 원고를 다시 생성하지 않고 PDF 두 개와 검증 기록만 추가한다.
 * --pdf-backend powerpoint는 Windows에 설치된 PowerPoint의 공식 PDF 내보내기를
 * 명시적으로 선택한다. 기본 artifact-tool 변환 실패를 자동으로 숨기지는 않는다.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');	// 현재 실행 위치와 무관하게 저장소 루트를 계산한다.
const SOURCE_COMMIT = '2edda38ac1c6a177cf7496cdf9c2550ecc446b7b';	// 수치와 설명의 근거를 검증 완료된 커밋에 고정한다.
const REPOSITORY = 'https://github.com/ntt65/A2_1';
const FONT = 'Malgun Gothic';
const WIDTH = 1600, HEIGHT = 900, MARGIN = 72, CONTENT_WIDTH = WIDTH - 2 * MARGIN;	// 16:9 화면과 양쪽 여백을 픽셀 단위로 정의한다.
const COLORS = {ink: '#132E48', teal: '#087F82', muted: '#536474', line: '#CCD6DF', white: '#FFFFFF'};
const INPUTS = {
  linear: 'linear_algebra_metrics.json', calculus: 'calculus_metrics.json',
  backprop: 'backprop_metrics.json', optimizer: 'optimizer_metrics.json',
  probability: 'probability_metrics.json', bonus: 'bonus_optimizer_metrics.json',
  information: 'information_theory.json', sensitivity: 'optimizer_sensitivity.json',
};
const DECKS = [
  {module: 'mission_content.mjs', exportName: 'missionSlides', stem: 'a2_1_mission_implementation'},
  {module: 'qa_content.mjs', exportName: 'qaSlides', stem: 'a2_1_evaluation_qa'},
];

/** 경로 인자와 제작 단계를 해석한다. 누락·중복·상충 옵션은 오류로 거부한다. */
function parseOptions(argv) {
  const result = {};
  const names = new Set(['modules', 'skill-dir', 'python', 'output-dir', 'build-dir', 'pdf-backend']);
  for (let i = 0; i < argv.length; i++) {
    const argument = argv[i];
    if (argument === '--skip-pdf') {
      result.skipPdf = true;
    } else if (argument === '--pdf-only') {
      result.pdfOnly = true;
    } else if (argument === '--help') {
      result.help = true;
    } else {
      const name = argument.startsWith('--') ? argument.slice(2) : '';
      if (!names.has(name) || !argv[i + 1] || argv[i + 1].startsWith('--')) {
        throw new Error(`Unknown or incomplete option: ${argument}`);
      }
      if (result[name] !== undefined) throw new Error(`Duplicate option: ${argument}`);
      result[name] = argv[++i];
    }
  }
  if (result.help) return result;
  if (result.skipPdf && result.pdfOnly) throw new Error('--skip-pdf and --pdf-only cannot be used together');
  result.pdfBackend = result['pdf-backend'] || 'artifact-tool';
  if (!['artifact-tool', 'powerpoint'].includes(result.pdfBackend)) throw new Error('--pdf-backend must be artifact-tool or powerpoint');
  for (const name of ['modules', 'skill-dir', 'python']) {
    if (!result[name]) throw new Error(`Required option --${name} is missing`);
    result[name] = path.resolve(result[name]);
  }
  const revision = new Date().toISOString().replace(/[:.]/g, '-');	// 기본 빌드 폴더는 시각마다 구분하여 기존 검증 기록을 보존한다.
  result.output = path.resolve(ROOT, result['output-dir'] || 'docs/presentation');
  result.build = path.resolve(ROOT, result['build-dir'] || `.cache/presentation-build/${revision}`);
  return result;
}

/** 대상 경로가 지정한 작업 폴더의 하위인지 검사하며, 외부·상위 경로는 거부한다. */
function requireWithin(target, base, label) {
  const relative = path.relative(base, target);	// 문자열 접두사 대신 정규화된 상대 경로로 경계를 확인한다.
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error(`${label} must be a child of ${base}: ${target}`);
  }
}

/** 파일 존재 여부를 반환한다. 파일 없음 이외의 파일 시스템 오류는 전달한다. */
async function exists(filename) {
  try { await fs.access(filename); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

/** 입력 또는 산출물 전체 바이트의 SHA-256을 계산하여 동일성 근거로 남긴다. */
function hash(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');	// 실행 세션에 따라 달라지지 않는 전체 바이트 해시를 사용한다.
}

/** 검증 객체를 읽기 쉬운 JSON 새 파일로 저장한다. 기존 파일은 덮어쓰지 않는다. */
async function writeJson(filename, value) {
  await fs.writeFile(filename, `${JSON.stringify(value, null, 2)}\n`, {encoding: 'utf8', flag: 'wx'});	// 배타적 생성으로 기존 기록을 실수로 덮어쓰지 않는다.
}

/** 내보내기 Blob을 변형 없이 새 파일에 저장하고 경로·크기·해시를 반환한다. */
async function writeBlob(filename, blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());	// 내보내기 결과의 원본 바이트를 그대로 읽는다.
  await fs.writeFile(filename, bytes, {flag: 'wx'});	// 완성 산출물은 새 경로에만 기록한다.
  return {path: filename, bytes: bytes.length, sha256: hash(bytes)};
}

/** 기존 실험 JSON 여덟 개를 읽어 원고 함수에 전달할 별칭과 입력 해시를 만든다. */
async function loadMeasurements() {
  const data = {}, provenance = [];
  for (const [key, filename] of Object.entries(INPUTS)) {
    const fullPath = path.join(ROOT, 'outputs', filename);
    const bytes = await fs.readFile(fullPath);
    data[key] = JSON.parse(bytes.toString('utf8'));	// 기존 실측을 읽으며 실험을 다시 실행하거나 값을 바꾸지 않는다.
    provenance.push({key, path: `outputs/${filename}`, sha256: hash(bytes)});	// 원고에 사용된 실측 파일의 동일성을 기록한다.
  }
  data.linearAlgebra = data.linear;
  data.bonusOptimizer = data.bonus;
  data.informationTheory = data.information;
  return {data, provenance};
}

/** 원고 배열의 필수 문자열·고유 ID·이미지·표 형상을 검사하고 불완전한 원고를 거부한다. */
function validateSpecifications(specifications, deckName) {
  if (!Array.isArray(specifications) || specifications.length === 0) {
    throw new Error(`${deckName}: content function must return a nonempty slide array`);
  }
  const ids = new Set();
  for (const spec of specifications) {
    if (!spec || typeof spec !== 'object') throw new TypeError(`${deckName}: invalid slide object`);
    for (const key of ['id', 'title', 'notes']) {
      if (typeof spec[key] !== 'string' || !spec[key].trim()) throw new Error(`${deckName}: missing ${key}`);
    }
    if (ids.has(spec.id)) throw new Error(`${deckName}: duplicate slide id ${spec.id}`);
    ids.add(spec.id);	// 페이지 순서와 별개인 원고 ID의 중복을 막는다.
    if (!['cover', 'text', 'table', 'image', 'formula'].includes(spec.kind)) {
      throw new Error(`${spec.id}: unsupported kind ${spec.kind}`);
    }
    for (const key of ['lines', 'formula', 'sources']) {
      if (spec[key] !== undefined && (!Array.isArray(spec[key]) || spec[key].some(item => typeof item !== 'string'))) {
        throw new TypeError(`${spec.id}: ${key} must contain strings`);
      }
    }
    if (!spec.sources?.length) throw new Error(`${spec.id}: at least one source is required`);
    for (const key of ['lead', 'interpretation']) {
      if (spec[key] !== undefined && typeof spec[key] !== 'string') throw new TypeError(`${spec.id}: invalid ${key}`);
    }
    if (spec.image && spec.images) throw new Error(`${spec.id}: provide image or images, not both`);
    const images = spec.images || (spec.image ? [spec.image] : []);
    if (!Array.isArray(images) || images.length > 2 || images.some(item => typeof item !== 'string')) {
      throw new Error(`${spec.id}: use at most two existing PNG image paths`);
    }
    if (spec.kind === 'cover' && (images.length || spec.table || spec.formula?.length)) {
      throw new Error(`${spec.id}: cover must contain only its title, lead and short lines`);
    }
    if (spec.table) {
      const {headers, rows, weights} = spec.table;
      if (!Array.isArray(headers) || headers.length < 2 || !Array.isArray(rows) || !rows.length || rows.length > 13) {
        throw new Error(`${spec.id}: a table needs >=2 columns and 1..13 body rows`);
      }
      for (const row of [headers, ...rows]) {
        if (!Array.isArray(row) || row.length !== headers.length || row.some(value => !['string', 'number'].includes(typeof value))) {
          throw new Error(`${spec.id}: table rows must match the header without missing cells`);
        }
        if (row.some(value => typeof value === 'number' && !Number.isFinite(value))) {
          throw new Error(`${spec.id}: table contains a nonfinite number; author its boundary label explicitly`);
        }
      }
      if (weights && (weights.length !== headers.length || weights.some(value => !Number.isFinite(value) || value <= 0))) {
        throw new Error(`${spec.id}: column weights must be positive and match the header`);
      }
    }
  }
}

/** 저장소 근거를 고정 커밋 URL로 변환한다. 외부 URL과 유효한 줄 번호는 보존한다. */
function sourceUrl(reference) {
  if (/^https?:\/\//i.test(reference)) return reference;
  const normalized = reference.replaceAll('\\', '/').replace(/^\.\//, '');
  const [filename, fragment] = normalized.split('#', 2);
  if (!filename || filename.startsWith('/') || filename.split('/').includes('..')) {
    throw new Error(`Invalid repository source: ${reference}`);
  }
  const encoded = filename.split('/').map(encodeURIComponent).join('/');	// 한글 파일명과 공백이 포함된 근거 경로도 URL로 표현한다.
  const anchor = fragment && /^L\d+(?:-L\d+)?$/.test(fragment) ? `#${fragment}` : '';
  return `${REPOSITORY}/blob/${SOURCE_COMMIT}/${encoded}${anchor}`;
}

/** 상세 발표자 노트를 보존하고 아직 없는 근거 URL만 중복 없이 덧붙인다. */
function speakerNotes(spec) {
  const additions = [];
  const seen = new Set();
  for (const reference of spec.sources) {
    const url = sourceUrl(reference);
    if (!spec.notes.includes(url) && !seen.has(url)) additions.push(`${reference}\n${url}`);	// 원고 노트에 이미 있는 근거는 반복하지 않는다.
    seen.add(url);
  }
  return [spec.notes, additions.length ? `근거\n${additions.join('\n\n')}` : ''].filter(Boolean).join('\n\n');
}

/** 설치된 맑은 고딕의 실제 글자 폭을 측정하는 도구를 만든다. 비트맵은 생성하지 않는다. */
function makeTypography(requireRuntime) {
  const {createCanvas, GlobalFonts} = requireRuntime('@napi-rs/canvas');
  if (!GlobalFonts.has(FONT)) {
    const fonts = path.join(process.env.WINDIR || 'C:/Windows', 'Fonts');
    if (!GlobalFonts.registerFromPath(path.join(fonts, 'malgun.ttf'), FONT)) {
      throw new Error(`Required font is unavailable: ${FONT}`);
    }
    GlobalFonts.registerFromPath(path.join(fonts, 'malgunbd.ttf'), FONT);
  }
  const context = createCanvas(8, 8).getContext('2d');	// 도형을 그리지 않고 설치 글꼴의 문자열 폭만 측정한다.
  /** 슬라이드와 같은 글꼴·픽셀 크기·굵기로 문자열의 폭과 글꼴 높이를 측정한다. */
  function measure(value, size, bold = false) {
    context.font = `${bold ? 'bold ' : ''}${size}px "${FONT}"`;	// 실제 슬라이드에 적용할 크기와 굵기를 그대로 사용한다.
    return context.measureText(value);
  }
  /** 공백을 우선하여 줄을 나누고, 긴 단어는 유니코드 문자 경계에서 나눈다. */
  function wrap(value, width, size, bold = false) {
    if (!value) return {text: '', lines: [], height: 0};
    if (width < size * 3) throw new Error(`Text frame is too narrow: ${width}px`);
    const result = [];
    for (const paragraph of value.replaceAll('\r\n', '\n').split('\n')) {
      if (!paragraph) { result.push(''); continue; }
      let line = '';
      for (const character of paragraph) {
        const trial = line + character;
        if (measure(trial, size, bold).width > width - 10 && line) {
          const breakAt = line.lastIndexOf(' ');
          if (breakAt > line.length / 3) {
            result.push(line.slice(0, breakAt));
            line = line.slice(breakAt + 1) + character;
          } else { result.push(line); line = character; }
        } else line = trial;
      }
      result.push(line.trimEnd());
    }
    const metrics = measure('가나다 Agβ∑√', size, bold);
    const lineHeight = Math.max(size * 1.27, metrics.fontBoundingBoxAscent + metrics.fontBoundingBoxDescent);	// 한글과 수식 기호가 잘리지 않도록 실측 높이에 여유를 둔다.
    return {text: result.join('\n'), lines: result, height: Math.ceil(result.length * lineHeight + 6), lineHeight};
  }
  return {wrap, measure};
}

/** 측정된 높이의 편집 가능한 텍스트를 추가한다. 영역 부족 시 축소하지 않고 오류를 낸다. */
function addText(slide, value, box, typography, records, options = {}) {
  const {size = 28, color = COLORS.ink, bold = false, alignment = 'left', label = 'text'} = options;
  if (size < 24) throw new Error(`${label}: native text below the 24px design minimum`);
  const wrapped = typography.wrap(value, box.width, size, bold);	// 렌더 전에 필요한 줄 수와 높이를 확인한다.
  if (wrapped.height > box.height + 0.5) {
    throw new Error(`${label}: ${wrapped.lines.length} lines need ${wrapped.height}px, only ${box.height}px available. Split the slide; do not shrink text.`);
  }
  if (!value) return null;
  const shape = slide.shapes.add({geometry: 'textbox', name: label,
    position: box, fill: 'none', line: {fill: 'none', width: 0}});
  shape.text = wrapped.text;	// 내용을 이미지로 굳히지 않고 편집 가능한 문자열로 넣는다.
  shape.text.style = {typeface: FONT, fontSize: size, color, bold, alignment,
    verticalAlignment: 'top', wrap: 'square', autoFit: 'none', insets: 0};	// 자동 글자 축소를 금지하고 측정한 배치를 유지한다.
  records.push({kind: 'text', label, text: value, wrappedText: wrapped.text, fontSizePx: size, ...box});
  return shape;
}

/** 수식 다음 본문 순서로 각 문자열을 측정하여 줄별 높이와 총 필요 높이를 반환한다. */
function narrativeLayout(spec, width, typography) {
  const blocks = [];
  for (const [kind, values, size] of [['formula', spec.formula || [], 30], ['body', spec.lines || [], 28]]) {
    for (const [index, value] of values.entries()) {
      if (!value.trim()) continue;
      const measurement = typography.wrap(value, width, size);	// 수식과 본문을 각각 지정한 글자 크기로 측정한다.
      blocks.push({value, size, height: measurement.height, kind, index});
    }
  }
  return {blocks, height: blocks.reduce((sum, block) => sum + block.height + 10, 0)};
}

/** 미리 측정한 수식·본문을 순서대로 배치하고 다음 내용의 시작 y 좌표를 반환한다. */
function addNarrative(slide, plan, x, y, width, typography, records, id) {
  for (const block of plan.blocks) {
    addText(slide, block.value, {left: x, top: y, width, height: block.height}, typography, records,
      {size: block.size, color: block.kind === 'formula' ? COLORS.teal : COLORS.ink, label: `${id}-${block.kind}-${block.index}`});
    y += block.height + 10;	// 앞 블록의 실측 높이와 일정한 문단 간격만큼 내려간다.
  }
  return y;
}

/** 표 열 가중치로 폭을 나누고, 24px 글자를 보존하는 열별 줄바꿈·행별 높이를 계산한다. */
function tableLayout(spec, width, typography) {
  if (!spec.table) return null;
  const {headers, rows} = spec.table;
  const weights = spec.table.weights || headers.map(() => 1);
  const total = weights.reduce((sum, value) => sum + value, 0);
  const widths = weights.map(value => value / total * width);	// 열 가중치의 합이 전체 표 너비가 되도록 비례 배분한다.
  const values = [headers, ...rows].map(row => row.map(String));
  const wrapped = values.map((row, r) => row.map((cell, c) => typography.wrap(cell, widths[c] - 24, 24, r === 0)));
  const heights = wrapped.map(row => Math.max(...row.map(cell => cell.height)) + 8);	// 한 행에서 가장 긴 셀을 기준으로 높이를 보존한다.
  return {values: wrapped.map(row => row.map(cell => cell.text)), widths, heights,
    height: heights.reduce((sum, value) => sum + value, 0)};
}

/** 셀을 직접 편집할 수 있는 PowerPoint 표를 측정된 크기와 최소 구분선으로 추가한다. */
function addTable(slide, plan, x, y, width, records, id) {
  const table = slide.tables.add({rows: plan.values.length, columns: plan.widths.length,
    left: x, top: y, width, height: plan.height, columnWidths: plan.widths, values: plan.values});	// 기본 표 객체를 사용하여 모든 셀을 편집 가능하게 만든다.
  table.borders.assign({style: 'solid', fill: COLORS.line, width: 0.75});
  table.cells.block({row: 0, column: 0, rowCount: plan.values.length, columnCount: plan.widths.length})
    .assign({margins: {left: 10, right: 10, top: 4, bottom: 4}});
  for (let r = 0; r < plan.values.length; r++) {
    table.rows[r].height = plan.heights[r];	// 줄바꿈에 필요한 높이를 각 행에 따로 적용한다.
    for (let c = 0; c < plan.widths.length; c++) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? COLORS.ink : COLORS.white;
      cell.text.style = {typeface: FONT, fontSize: 24, bold: r === 0,
        color: r === 0 ? COLORS.white : COLORS.ink, alignment: 'left',
        verticalAlignment: 'middle', autoFit: 'none', wrap: 'square'};
    }
  }
  records.push({kind: 'native-table', label: `${id}-table`, rows: plan.values.length,
    columns: plan.widths.length, values: plan.values, fontSizePx: 24,
    left: x, top: y, width, height: plan.height, rowHeights: plan.heights, columnWidths: plan.widths});
}

/** outputs 하위의 기존 PNG를 읽어 원래 가로·세로 크기와 바이트 해시를 보존한다. */
async function readEvidenceImage(reference) {
  const filename = path.resolve(ROOT, reference);
  requireWithin(filename, path.join(ROOT, 'outputs'), 'Evidence image');	// 기존 실험 출력 그림만 삽입한다.
  const bytes = await fs.readFile(filename);
  if (bytes.length < 24 || bytes.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a') {
    throw new Error(`Evidence image is not a PNG: ${reference}`);
  }
  return {filename, reference, bytes, width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20), sha256: hash(bytes)};
}

/** 원본 PNG를 바꾸지 않고 종횡비를 유지하여 주어진 영역 중앙에 삽입한다. */
function addImage(slide, image, frame, records) {
  const scale = Math.min(frame.width / image.width, frame.height / image.height);	// 가로·세로 중 작은 배율을 사용하여 그림을 자르거나 늘이지 않는다.
  const width = image.width * scale, height = image.height * scale;
  const position = {left: frame.left + (frame.width - width) / 2,
    top: frame.top + (frame.height - height) / 2, width, height};
  slide.images.add({blob: image.bytes.buffer.slice(image.bytes.byteOffset, image.bytes.byteOffset + image.bytes.byteLength),
    contentType: 'image/png', alt: image.reference, fit: 'contain', position});
  records.push({kind: 'image', source: image.reference, sha256: image.sha256,
    sourceWidth: image.width, sourceHeight: image.height, ...position});
}

/** 원고의 모든 필드를 배치한다. 공간 부족 시 해당 슬라이드 ID와 수정 이유를 보고한다. */
async function addSlide(presentation, spec, index, count, typography) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.white;
  const records = [];
  const images = await Promise.all((spec.images || (spec.image ? [spec.image] : [])).map(readEvidenceImage));	// 서로 독립된 원본 그림 읽기를 함께 수행한다.
  const titleSize = spec.kind === 'cover' ? 56 : 46;	// 표지 제목 42pt와 본문 제목 34.5pt를 유지한다.
  const title = typography.wrap(spec.title, CONTENT_WIDTH, titleSize, true);
  if (title.lines.length > 2) throw new Error(`${spec.id}: title needs ${title.lines.length} lines; shorten the title`);
  let y = spec.kind === 'cover' ? 220 : 42;
  addText(slide, spec.title, {left: MARGIN, top: y, width: CONTENT_WIDTH, height: title.height}, typography, records,
    {size: titleSize, bold: true, label: `${spec.id}-title`});
  y += title.height + (spec.kind === 'cover' ? 34 : 15);
  if (spec.lead) {
    const lead = typography.wrap(spec.lead, CONTENT_WIDTH, 28);
    if (lead.lines.length > 2) throw new Error(`${spec.id}: lead exceeds two lines; move details into the body or notes`);
    addText(slide, spec.lead, {left: MARGIN, top: y, width: CONTENT_WIDTH, height: lead.height}, typography, records,
      {size: 28, color: COLORS.muted, label: `${spec.id}-lead`});
    y += lead.height + 22;
  }
  y = Math.max(y, spec.kind === 'cover' ? 450 : 178);
  const interpretation = typography.wrap(spec.interpretation || '', CONTENT_WIDTH, 28);
  if (interpretation.lines.length > 3) throw new Error(`${spec.id}: interpretation exceeds three lines; split the explanation`);
  const bottom = 832 - (interpretation.height ? interpretation.height + 18 : 0);
  const available = bottom - y;	// 제목·요약·해석을 제외한 실제 본문 높이를 계산한다.
  const narrative = narrativeLayout(spec, CONTENT_WIDTH, typography);
  const table = tableLayout(spec, CONTENT_WIDTH, typography);
  if (!images.length) {
    const required = narrative.height + (table ? table.height + (narrative.height ? 10 : 0) : 0);	// 텍스트와 표가 함께 있을 때 필요한 높이를 합산한다.
    if (required > available) throw new Error(`${spec.id}: body/table need ${Math.ceil(required)}px but only ${Math.floor(available)}px remain. Split this slide.`);
    y = addNarrative(slide, narrative, MARGIN, y, CONTENT_WIDTH, typography, records, spec.id);
    if (table) addTable(slide, table, MARGIN, y + (narrative.height ? 10 : 0), CONTENT_WIDTH, records, spec.id);
  } else if (images.length === 1 && table && spec.table.rows.length <= 5 && images[0].width / images[0].height < 1.65) {
    const imageWidth = 740, gap = 36, textWidth = CONTENT_WIDTH - imageWidth - gap;	// 세로에 가까운 그림과 짧은 표는 좌우로 나누어 그림을 크게 보존한다.
    const rightNarrative = narrativeLayout(spec, textWidth, typography);
    const rightTable = tableLayout(spec, textWidth, typography);
    const required = rightNarrative.height + (rightNarrative.height ? 10 : 0) + rightTable.height;
    if (required > available) throw new Error(`${spec.id}: image-side text/table need ${Math.ceil(required)}px, only ${Math.floor(available)}px remain. Split this slide.`);
    addImage(slide, images[0], {left: MARGIN, top: y, width: imageWidth, height: available}, records);
    const tableY = addNarrative(slide, rightNarrative, MARGIN + imageWidth + gap, y, textWidth, typography, records, spec.id);
    addTable(slide, rightTable, MARGIN + imageWidth + gap, tableY + (rightNarrative.height ? 10 : 0), textWidth, records, spec.id);
  } else if (images.length === 1 && !table && narrative.blocks.length && images[0].width / images[0].height < 1.65) {
    const imageWidth = 870, gap = 36, textWidth = CONTENT_WIDTH - imageWidth - gap;
    const right = narrativeLayout(spec, textWidth, typography);
    if (right.height > available) throw new Error(`${spec.id}: image-side explanation needs ${right.height}px, only ${available}px remain`);
    addImage(slide, images[0], {left: MARGIN, top: y, width: imageWidth, height: available}, records);
    addNarrative(slide, right, MARGIN + imageWidth + gap, y, textWidth, typography, records, spec.id);
  } else {
    const imageGap = 18;
    const imageHeight = available - narrative.height - (table ? table.height + imageGap : 0) - (narrative.height ? imageGap : 0);	// 글자와 표를 보존한 나머지 공간을 그림에 배정한다.
    if (imageHeight < 200) throw new Error(`${spec.id}: PNG would have only ${Math.floor(imageHeight)}px height. Split text/table onto another slide.`);
    y = addNarrative(slide, narrative, MARGIN, y, CONTENT_WIDTH, typography, records, spec.id);
    if (narrative.height) y += imageGap;
    const gap = images.length === 2 ? 28 : 0;
    const width = (CONTENT_WIDTH - gap) / images.length;
    images.forEach((image, imageIndex) => addImage(slide, image,
      {left: MARGIN + imageIndex * (width + gap), top: y, width, height: imageHeight}, records));
    if (table) addTable(slide, table, MARGIN, y + imageHeight + imageGap, CONTENT_WIDTH, records, spec.id);
  }
  if (spec.interpretation) addText(slide, spec.interpretation,
    {left: MARGIN, top: 832 - interpretation.height, width: CONTENT_WIDTH, height: interpretation.height}, typography, records,
    {size: 28, color: COLORS.teal, label: `${spec.id}-interpretation`});
  addText(slide, `${index + 1} / ${count}`, {left: WIDTH - MARGIN - 180, top: 852, width: 180, height: 42}, typography, records,
    {size: 24, color: COLORS.muted, alignment: 'right', label: `${spec.id}-page`});
  slide.speakerNotes.text = speakerNotes(spec);	// 화면에 다 담지 않은 상세 설명과 고정 근거를 발표자 노트에 보존한다.
  return {number: index + 1, id: spec.id, title: spec.title, kind: spec.kind,
    sources: spec.sources.map(reference => ({reference, url: sourceUrl(reference)})), records,
    notesCharacters: spec.notes.length, hasNativeTable: Boolean(table), formulaCount: spec.formula?.length || 0};
}

/** 최종 PPTX XML의 글꼴 크기·비어 있지 않은 본문·편집 가능한 표 개수를 확인한다. */
function inspectPptx(python, filename, expectedCount) {
  const program = `import json,sys,zipfile,xml.etree.ElementTree as E,re\n` +
    `z=zipfile.ZipFile(sys.argv[1]); ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}\n` +
    `names=sorted((n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide[0-9]+.xml',n)),key=lambda n:int(re.search(r'slide([0-9]+)',n).group(1)))\n` +
    `assert len(names)==int(sys.argv[2]), 'PPTX slide count mismatch'\n` +
    `pages=[]\n` +
    `for name in names:\n` +
    ` root=E.fromstring(z.read(name)); sizes=[int(n.attrib['sz'])/100 for n in root.iter() if n.tag.endswith(('}rPr','}defRPr')) and 'sz' in n.attrib]\n` +
    ` assert sizes and min(sizes)>=17, 'Native slide font below 17pt: '+name\n` +
    ` texts=[n.text or '' for n in root.findall('.//a:t',ns)]; assert any(t.strip() for t in texts), 'Empty slide: '+name\n` +
    ` pages.append({'part':name,'minimumFontPt':min(sizes),'tableCount':len(root.findall('.//a:tbl',ns)),'textCharacters':sum(map(len,texts))})\n` +
    `print(json.dumps({'slideCount':len(names),'pages':pages},ensure_ascii=False))\n`;
  return JSON.parse(execFileSync(python, ['-X', 'utf8', '-c', program, filename, String(expectedCount)],
    {encoding: 'utf8', windowsHide: true, maxBuffer: 4 * 1024 * 1024}));
}

/** PDF 페이지 수와 본문 존재를 확인한다. 텍스트 추출 결과와 시각 검토 필요성은 따로 기록한다. */
function inspectPdf(python, filename, specifications) {
  const program = `import json,sys,re\nfrom pypdf import PdfReader\n` +
    `reader=PdfReader(sys.argv[1]); specs=json.load(sys.stdin)\n` +
    `assert len(reader.pages)==len(specs), 'PDF page count mismatch'\n` +
    `pages=[]\n` +
    `for page,spec in zip(reader.pages,specs):\n` +
    ` text=page.extract_text() or ''; images=len(page.images); contents=page.get_contents(); raw=contents.get_data() if contents is not None else b''\n` +
    ` assert raw and (text.strip() or images), 'Empty PDF page: '+spec['id']\n` +
    ` normalized=re.sub(r'\\s+','',text); title=re.sub(r'\\s+','',spec['title']); match=title in normalized if text.strip() else None\n` +
    ` pages.append({'id':spec['id'],'textCharacters':len(text),'imageCount':images,'titleExtracted':match,'widthPt':float(page.mediabox.width),'heightPt':float(page.mediabox.height)})\n` +
    `print(json.dumps({'pageCount':len(pages),'pages':pages,'visualReviewRequired':True},ensure_ascii=False))\n`;
  return JSON.parse(execFileSync(python, ['-X', 'utf8', '-c', program, filename],
    {input: JSON.stringify(specifications.map(({id, title}) => ({id, title}))), encoding: 'utf8',
      windowsHide: true, maxBuffer: 8 * 1024 * 1024}));
}

/** 지정된 Python과 함께 제공된 Poppler 실행 파일을 찾고, 없으면 오류를 낸다. */
async function popplerExecutable(python) {
  const dependencies = path.resolve(path.dirname(python), '..');
  const filename = path.join(dependencies, 'native/poppler/Library/bin/pdftoppm.exe');
  if (!await exists(filename)) throw new Error(`Bundled PDF renderer unavailable: ${filename}`);
  return filename;
}

/** Windows PowerPoint에서 최종 PPTX를 읽기 전용으로 열어 공식 API로 PDF를 내보낸다. */
async function exportPowerPointPdf(pptxPath, pdfPath, directory) {
  if (process.platform !== 'win32') throw new Error('The powerpoint PDF backend requires Windows and installed Microsoft PowerPoint');
  if (await exists(pdfPath)) throw new Error(`Refusing to overwrite ${pdfPath}`);
  const beforeHash = hash(await fs.readFile(pptxPath));	// 변환 전후 원본 PPTX가 동일한지 별도로 확인한다.
  const scriptPath = path.join(directory, 'export-powerpoint.ps1');
  const script = `param([Parameter(Mandatory=$true)][string]$PptxPath,[Parameter(Mandatory=$true)][string]$PdfPath)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$application = $null
$presentation = $null
$printRange = $null
$createdApplication = $false
try {
  if (Test-Path -LiteralPath $PdfPath) { throw "PDF already exists: $PdfPath" }
  try { $application = [Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application') }
  catch { $application = New-Object -ComObject PowerPoint.Application; $createdApplication = $true }
  $presentation = $application.Presentations.Open($PptxPath, -1, 0, 0)	# 읽기 전용이며 창을 표시하지 않고 연다.
  $count = $presentation.Slides.Count
  $version = $application.Version
  $printRange = $presentation.PrintOptions.Ranges.Add(1, $count)	# PowerShell COM 바인딩에는 Nothing 대신 실제 전체 범위 객체를 전달한다.
  $presentation.ExportAsFixedFormat($PdfPath, 2, 2, 0, 1, 1, 0, $printRange, 1, '', $false, $true, $true, $true, $false)	# PDF·인쇄 품질·전체 슬라이드를 지정한다.
  [PSCustomObject]@{backend='Microsoft PowerPoint';version=$version;slideCount=$count;method='Presentation.ExportAsFixedFormat';readOnly=$true} | ConvertTo-Json -Compress
} finally {
  if ($null -ne $printRange) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($printRange) }
  if ($null -ne $presentation) { $presentation.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) }
  if ($null -ne $application) {
    if ($createdApplication -and $application.Presentations.Count -eq 0) { $application.Quit() }	# 다른 문서가 열린 앱은 종료하지 않는다.
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($application)
  }
}
`;
  await fs.writeFile(scriptPath, `\ufeff${script}`, {encoding: 'utf8', flag: 'wx'});	// 변환에 사용한 정확한 명령을 비공개 검증 폴더에 보존한다.
  const renderer = JSON.parse(execFileSync('powershell.exe', ['-NoLogo', '-NoProfile', '-NonInteractive', '-File', scriptPath,
    '-PptxPath', pptxPath, '-PdfPath', pdfPath], {encoding: 'utf8', windowsHide: true, maxBuffer: 4 * 1024 * 1024}));
  if (hash(await fs.readFile(pptxPath)) !== beforeHash) throw new Error('PowerPoint PDF export changed the source PPTX bytes');
  const bytes = await fs.readFile(pdfPath);
  return {path: pdfPath, bytes: bytes.length, sha256: hash(bytes), renderer,
    sourcePptxSha256: beforeHash, sourcePptxUnchanged: true,
    apiReference: 'https://learn.microsoft.com/en-us/office/vba/api/powerpoint.presentation.exportasfixedformat'};
}

/** 선택한 변환기로 최종 PPTX의 PDF를 만들고 동일한 전체 페이지·본문 검사를 적용한다. */
async function exportPdf(delivered, specifications, directory, finalPath, options, pptxPath) {
  const result = options.pdfBackend === 'powerpoint'
    ? await exportPowerPointPdf(pptxPath, finalPath, directory)
    : await writeBlob(finalPath, await delivered.export({format: 'pdf'}));	// 선택된 변환기만 사용하며 실패 시 자동으로 다른 결과를 만들지 않는다.
  result.backend = options.pdfBackend;
  if (result.renderer && result.renderer.slideCount !== specifications.length) throw new Error('PowerPoint opened an unexpected slide count');
  const inspection = inspectPdf(options.python, finalPath, specifications);	// 페이지 수와 본문 존재를 실제 PDF에서 확인한다.
  await writeJson(path.join(directory, 'pdf-inspection.json'), inspection);
  const pdfRenderDir = path.join(directory, 'pdf-pages');
  await fs.mkdir(pdfRenderDir);
  const poppler = await popplerExecutable(options.python);
  execFileSync(poppler, ['-png', '-r', '96', finalPath, path.join(pdfRenderDir, 'page')],
    {windowsHide: true, stdio: 'pipe', maxBuffer: 4 * 1024 * 1024});	// PDF의 모든 페이지를 독립된 렌더러로 확인할 수 있게 저장한다.
  const pdfPages = (await fs.readdir(pdfRenderDir)).filter(name => name.endsWith('.png'));
  if (pdfPages.length !== specifications.length) throw new Error(`${finalPath}: PDF rendering produced the wrong page count`);
  result.pageCount = inspection.pageCount;
  result.renderedPages = pdfPages.length;
  result.titleExtractionComplete = inspection.pages.every(page => page.titleExtracted === true);
  result.visualReviewRequired = true;	// 자동 본문 검사를 시각 검토 완료로 표현하지 않는다.
  return result;
}

/** 이미 존재하는 검증 JSON만 원자적으로 갱신한다. PPTX/PDF 완성 파일에는 사용하지 않는다. */
async function replaceValidationJson(filename, value) {
  if (!await exists(filename)) throw new Error(`Validation record is missing: ${filename}`);
  const temporary = `${filename}.${crypto.randomUUID()}.tmp`;
  await writeJson(temporary, value);	// 기존 기록을 유지한 채 같은 폴더의 임시 파일을 완성한다.
  await fs.rename(temporary, filename);	// 완성된 검증 JSON으로 교체한다.
}

/** 초안 빌드의 최종 PPTX 해시를 확인한 뒤 PDF만 추가한다. 원고·실측·PPTX는 재생성하지 않는다. */
async function finishPdfOnly(options, PresentationFile, FileBlob) {
  const manifestPath = path.join(options.build, 'manifest.json');
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  if (manifest.outputComplete || !Array.isArray(manifest.decks) || manifest.decks.length !== DECKS.length) {
    throw new Error('PDF-only requires an incomplete two-deck manifest produced by --skip-pdf');
  }
  const jobs = [];
  for (const config of DECKS) {
    const previous = manifest.decks.find(deck => deck.stem === config.stem);
    if (!previous || previous.pdf?.status !== 'explicitly_skipped') throw new Error(`${config.stem}: PDF was not explicitly skipped`);
    const pptxPath = path.join(options.output, `${config.stem}.pptx`);
    if (path.resolve(previous.pptx.path) !== pptxPath) throw new Error(`${config.stem}: output directory differs from the original build`);
    const bytes = await fs.readFile(pptxPath);
    if (hash(bytes) !== previous.pptx.sha256) throw new Error(`${config.stem}: reviewed PPTX bytes differ from the recorded final file`);
    const pdfPath = path.join(options.output, `${config.stem}.pdf`);
    if (await exists(pdfPath)) throw new Error(`Refusing to overwrite ${pdfPath}`);
    const directory = path.join(options.build, config.stem);
    const map = JSON.parse(await fs.readFile(path.join(directory, 'slide-map.json'), 'utf8'));
    if (!Array.isArray(map) || map.length !== previous.slideCount || map.some(page => !page.id || !page.title)) {
      throw new Error(`${config.stem}: invalid or incomplete slide map`);
    }
    inspectPptx(options.python, pptxPath, previous.slideCount);	// PDF 변환 직전에도 최종 PPTX의 페이지·글꼴·본문을 검사한다.
    for (const target of ['pdf-inspection.json', 'pdf-pages']) {
      if (await exists(path.join(directory, target))) throw new Error(`${config.stem}: previous PDF verification output already exists`);
    }
    jobs.push({config, previous, pptxPath, pdfPath, directory, specifications: map});
  }
  for (const job of jobs) {
    const delivered = await PresentationFile.importPptx(await FileBlob.load(job.pptxPath));	// 검토된 최종 PPTX를 가져온다.
    if (delivered.slides.items.length !== job.previous.slideCount) throw new Error(`${job.config.stem}: import changed slide count`);
    job.previous.pdf = await exportPdf(delivered, job.specifications, job.directory, job.pdfPath, options, job.pptxPath);
    await replaceValidationJson(path.join(job.directory, 'manifest.json'), job.previous);
    console.log(`${job.config.stem}: PDF export and page verification complete`);
  }
  manifest.outputComplete = true;
  manifest.pdfCompletion = {mode: 'finalized PPTX only', backend: options.pdfBackend, completedAt: new Date().toISOString(),
    builderSha256: hash(await fs.readFile(fileURLToPath(import.meta.url)))};
  await replaceValidationJson(manifestPath, manifest);	// 두 PDF가 모두 생성·검증된 뒤에만 전체 완료를 기록한다.
  console.log(`PDF generation complete: ${options.output}`);
}

/** 발표자료 한 벌을 제작·최종화·재가져오기·전체 렌더하고 추적 가능한 검증 기록을 남긴다. */
async function buildDeck(config, specifications, context) {
  const {options, Presentation, PresentationFile, FileBlob, finalizePresentation, typography} = context;
  const directory = path.join(options.build, config.stem);
  await fs.mkdir(directory, {recursive: true});
  const finalRenderDir = path.join(directory, 'final-slides');
  await fs.mkdir(finalRenderDir);
  const presentation = Presentation.create({slideSize: {width: WIDTH, height: HEIGHT}});
  const map = [], layoutErrors = [];
  for (const [index, spec] of specifications.entries()) {
    try { map.push(await addSlide(presentation, spec, index, specifications.length, typography)); }
    catch (error) { layoutErrors.push(new Error(`${spec.id}: ${error.message}`, {cause: error})); }
  }
  if (layoutErrors.length) throw new AggregateError(layoutErrors,
    `${config.stem}: ${layoutErrors.length} slide layout errors\n${layoutErrors.map(error => error.message).join('\n')}`);	// 모든 배치 오류를 모아 완성 파일 내보내기 전에 보고한다.
  await writeJson(path.join(directory, 'slide-map.json'), map);
  const candidatePath = path.join(directory, 'candidate.pptx');
  const finalPath = path.join(options.output, `${config.stem}.pptx`);
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);	// 검사 전 초안은 비공개 빌드 폴더에만 저장한다.
  const tableOwners = map.filter(slide => slide.hasNativeTable).map(slide => slide.number);
  const finalization = await finalizePresentation({workspaceDir: ROOT, candidatePath, finalPath,
    pythonExecutable: options.python,
    integrityValidatorPath: path.join(options['skill-dir'], 'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath: path.join(options['skill-dir'], 'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs: ['--expected-slide-size-emu', '15240000,8572500', '--validate-bullet-geometry', '--validate-heading-fit',
      ...tableOwners.flatMap(number => ['--require-native-table-slide', String(number)])],
    explicitTotalSlideCount: specifications.length, requiredNativeTableOwnerSlides: tableOwners,
    requiredNativeChartOwnerSlides: [], fontPolicy: {basis: 'design', families: [FONT]},
    verifyArtifactToolImport: true, receiptPath: path.join(directory, 'finalizer-validation.json')});
  await writeJson(path.join(directory, 'finalizer-result.json'), finalization);
  const pptxInspection = inspectPptx(options.python, finalPath, specifications.length);	// 최종화가 끝난 실제 PPTX의 글자·표·페이지를 확인한다.
  await writeJson(path.join(directory, 'pptx-inspection.json'), pptxInspection);
  const delivered = await PresentationFile.importPptx(await FileBlob.load(finalPath));	// 원본 작성 객체 대신 전달할 최종 파일을 다시 가져온다.
  if (delivered.slides.items.length !== specifications.length) throw new Error(`${config.stem}: import changed slide count`);
  const renders = [];
  for (const [index, slide] of delivered.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(3, '0')}`;
    const image = await delivered.export({slide, format: 'png', scale: 1});	// 최종 파일의 모든 슬라이드를 하나씩 렌더하여 검토 대상으로 남긴다.
    const record = await writeBlob(path.join(finalRenderDir, `${stem}.png`), image);
    const layout = await slide.export({format: 'layout'});	// 최종 배치 좌표도 그림과 같은 페이지 이름으로 보존한다.
    await fs.writeFile(path.join(finalRenderDir, `${stem}.layout.json`), await layout.text(), {encoding: 'utf8', flag: 'wx'});
    renders.push({id: specifications[index].id, ...record});
    console.log(`${config.stem}: final slide ${index + 1}/${specifications.length}`);
  }
  const pptxBytes = await fs.readFile(finalPath);
  const result = {stem: config.stem, slideCount: specifications.length,
    pptx: {path: finalPath, bytes: pptxBytes.length, sha256: hash(pptxBytes)},
    formulaRepresentation: 'editable Unicode text', nativeTableOwnerSlides: tableOwners, renders};
  if (options.skipPdf) {
    result.pdf = {status: 'explicitly_skipped', complete: false};
    console.log(`${config.stem}: PDF deliberately skipped by --skip-pdf; delivery is incomplete.`);
  } else {
    const pdfPath = path.join(options.output, `${config.stem}.pdf`);
    result.pdf = await exportPdf(delivered, specifications, directory, pdfPath, options, finalPath);
  }
  await writeJson(path.join(directory, 'manifest.json'), result);
  return result;
}

/** 입력·출력 보호 검사가 끝난 뒤 두 발표자료 제작 또는 검토된 PPTX의 PDF 변환을 실행한다. */
async function main() {
  const options = parseOptions(process.argv.slice(2));
  if (options.help) {
    console.log('node build_presentations.mjs --modules PATH --skill-dir PATH --python PATH [--output-dir DIR] [--build-dir DIR] [--skip-pdf | --pdf-only] [--pdf-backend artifact-tool|powerpoint]');
    return;
  }
  requireWithin(options.output, ROOT, 'Output directory');
  requireWithin(options.build, ROOT, 'Build directory');
  if (options.output === options.build || options.build.startsWith(options.output + path.sep) || options.output.startsWith(options.build + path.sep)) {
    throw new Error('Build and output directories must be separate, non-nested directories');
  }
  if (options.pdfOnly) {
    const requireRuntime = createRequire(path.join(options.modules, 'a2-presentation-entry.cjs'));
    process.env.RUNTIME_NODE_MODULES = options.modules;
    const {PresentationFile, FileBlob} = await import(pathToFileURL(requireRuntime.resolve('@oai/artifact-tool')).href);
    await finishPdfOnly(options, PresentationFile, FileBlob);
    return;
  }
  for (const deck of DECKS) for (const extension of ['pptx', 'pdf']) {
    const target = path.join(options.output, `${deck.stem}.${extension}`);
    if (await exists(target)) throw new Error(`Refusing to overwrite ${target}; choose a new --output-dir revision`);
  }
  if (await exists(options.build) && (await fs.readdir(options.build)).length) {
    throw new Error(`Build directory is not empty: ${options.build}; choose a new revision`);
  }
  const {data, provenance} = await loadMeasurements();
  const decks = [];
  for (const config of DECKS) {
    const filename = path.join(ROOT, 'scripts/presentation', config.module);
    const module = await import(pathToFileURL(filename).href);
    if (typeof module[config.exportName] !== 'function') throw new Error(`Missing export ${config.exportName}: ${filename}`);
    const specifications = module[config.exportName](data);	// 저장된 실측 객체를 각 발표자료 원고 함수에 전달한다.
    validateSpecifications(specifications, config.stem);	// 제작을 시작하기 전에 필수 내용과 표 형상을 검사한다.
    decks.push({config, specifications, source: {path: filename, sha256: hash(await fs.readFile(filename))}});
  }
  const requireRuntime = createRequire(path.join(options.modules, 'a2-presentation-entry.cjs'));
  process.env.RUNTIME_NODE_MODULES = options.modules;
  const {Presentation, PresentationFile, FileBlob} = await import(pathToFileURL(requireRuntime.resolve('@oai/artifact-tool')).href);
  const {finalizePresentation} = await import(pathToFileURL(path.join(options['skill-dir'], 'container_tools/artifact_tool_utils.mjs')).href);
  const typography = makeTypography(requireRuntime);
  await fs.mkdir(options.build, {recursive: true});
  await fs.mkdir(options.output, {recursive: true});
  const context = {options, Presentation, PresentationFile, FileBlob, finalizePresentation, typography};
  const results = [];
  for (const {config, specifications} of decks) results.push(await buildDeck(config, specifications, context));
  const manifest = {sourceCommit: SOURCE_COMMIT, repository: REPOSITORY, fontFamily: FONT,
    slideSizePx: {width: WIDTH, height: HEIGHT}, minimumNativeFontPt: 17,
    authoringMarker: 'managed once by coordinating workflow; never rerun by builder',
    inputs: provenance, contentSources: decks.map(deck => deck.source),
    builder: {path: fileURLToPath(import.meta.url), sha256: hash(await fs.readFile(fileURLToPath(import.meta.url)))},
    outputComplete: !options.skipPdf, visualReviewRequired: true, decks: results};
  await writeJson(path.join(options.build, 'manifest.json'), manifest);	// 두 발표자료·입력·원고·빌더의 해시와 실제 완료 범위를 함께 기록한다.
  console.log(`Presentation build complete: ${options.output}`);
}

main().catch(error => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
