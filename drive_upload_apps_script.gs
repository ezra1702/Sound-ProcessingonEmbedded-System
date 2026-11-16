// Google Apps Script — penerima rekaman Perekam_Dataset_MP09.html -> Google Drive.
// Jalan di server Google, jadi komputer siapa pun tidak perlu menyala.
// Struktur di Drive:  MP09_Dataset/<Nama>/block_01..block_06/{nyala,mati,naik,turun}_bNN.wav
// Ganti SECRET di bawah dan di file HTML (harus sama).

const ROOT_NAME = 'MP09_Dataset';
const SECRET = 'GANTI-KODE-RAHASIA-KELOMPOK';
const MEMBERS = ['Christama Ezra Yudianto', 'Abdillah Fatah Suryanto', 'Ivang Arfandia', 'Jahzeel Erason Nicodemus'];
const NAME_RE = /^(nyala|mati|naik|turun)_b0([1-6])\.wav$/;

function doPost(e) {
  const lock = LockService.getScriptLock();
  try {
    const body = JSON.parse(e.postData.contents);
    if (body.secret !== SECRET) return out({ok: false, error: 'kode rahasia salah'});
    const m = NAME_RE.exec(body.name || '');
    if (!m) return out({ok: false, error: 'nama file tidak valid'});
    if (MEMBERS.indexOf(body.recorder) < 0) return out({ok: false, error: 'nama perekam tidak dikenal'});
    const bytes = Utilities.base64Decode(body.data);
    if (bytes.length < 1000 || bytes.length > 200000) return out({ok: false, error: 'ukuran file tidak wajar'});
    lock.waitLock(20000);
    const folder = child_(child_(root_(), body.recorder), 'block_0' + m[2]);
    const old = folder.getFilesByName(body.name);
    while (old.hasNext()) old.next().setTrashed(true);   // rekam ulang = ganti file lama
    const f = folder.createFile(Utilities.newBlob(bytes, 'audio/wav', body.name));
    f.setDescription(JSON.stringify({recorder: body.recorder, recognized_text: body.heard, recorded_at: body.at}));
    return out({ok: true, recorder: body.recorder, name: body.name, id: f.getId()});
  } catch (err) {
    return out({ok: false, error: String(err)});
  } finally {
    try { lock.releaseLock(); } catch (x) {}
  }
}

// GET ?secret=...                          -> daftar file per orang (progres di halaman)
// GET ?secret=...&recorder=...&file=...    -> isi satu rekaman (base64) untuk diputar ulang
function doGet(e) {
  const q = e.parameter || {};
  if (q.secret !== SECRET) return out({ok: false, error: 'kode rahasia salah'});
  if (q.file) {
    const m = NAME_RE.exec(q.file);
    if (!m || MEMBERS.indexOf(q.recorder) < 0) return out({ok: false, error: 'permintaan tidak valid'});
    const it = child_(child_(root_(), q.recorder), 'block_0' + m[2]).getFilesByName(q.file);
    if (!it.hasNext()) return out({ok: false, error: 'rekaman tidak ditemukan'});
    return out({ok: true, data: Utilities.base64Encode(it.next().getBlob().getBytes())});
  }
  const files = [], people = root_().getFolders();
  while (people.hasNext()) {
    const p = people.next();
    if (MEMBERS.indexOf(p.getName()) < 0) continue;
    const blocks = p.getFolders();
    while (blocks.hasNext()) {
      const it = blocks.next().getFiles();
      while (it.hasNext()) {
        const f = it.next();
        if (NAME_RE.test(f.getName())) files.push({recorder: p.getName(), name: f.getName()});
      }
    }
  }
  return out({ok: true, root: ROOT_NAME, files: files});
}

function root_() {
  const it = DriveApp.getFoldersByName(ROOT_NAME);
  return it.hasNext() ? it.next() : DriveApp.createFolder(ROOT_NAME);
}
function child_(parent, name) {
  const it = parent.getFoldersByName(name);
  return it.hasNext() ? it.next() : parent.createFolder(name);
}
function out(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}
