// NOTE: set this to your deployed backend (https://...) before production
const API_URL = 'http://localhost:8000';

const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const uploadForm = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const healthBtn = document.getElementById('healthBtn');
const statusText = document.getElementById('statusText');
const resultSection = document.getElementById('result');
const previewImg = document.getElementById('previewImg');
const resultImg = document.getElementById('resultImg');
const shapeText = document.getElementById('shapeText');
const dropZone = document.getElementById('dropZone');

let selectedFile = null;

function setStatus(text){ statusText.textContent = text; }

fileInput.addEventListener('change', (e) => {
  selectedFile = e.target.files[0] || null;
  fileName.textContent = selectedFile ? selectedFile.name : 'Choose or drop an image';
  if (selectedFile) previewFile(selectedFile);
});

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e)=>{ e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', ()=> dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e)=>{
  e.preventDefault(); dropZone.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f){ fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event('change')); }
});

function previewFile(file){
  const reader = new FileReader();
  reader.onload = ()=> previewImg.src = reader.result;
  reader.readAsDataURL(file);
}

uploadForm.addEventListener('submit', async (e)=>{
  e.preventDefault();
  if (!selectedFile) return setStatus('Please select an image first.');
  submitBtn.disabled = true;
  submitBtn.textContent = 'Processing…';
  setStatus('Uploading image and processing…');
  try {
    const fd = new FormData();
    fd.append('file', selectedFile);
    const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const data = await res.json();
    if (data.mask_png_b64){
      resultImg.src = `data:image/png;base64,${data.mask_png_b64}`;
      shapeText.textContent = `Output shape: ${JSON.stringify(data.shape)}`;
      resultSection.classList.remove('hidden');
      setStatus('Done — see result.');
    } else {
      setStatus('No output returned from API.');
    }
  } catch (err){
    setStatus('Error: ' + (err.message || err));
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Process Image';
  }
});

healthBtn.addEventListener('click', async ()=>{
  setStatus('Checking API health…');
  try {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) throw new Error(`${res.status}`);
    const info = await res.json();
    setStatus('API healthy — details below.');
    statusText.textContent = JSON.stringify(info, null, 2);
  } catch (err){
    setStatus('Health check failed: ' + (err.message || err));
  }
});