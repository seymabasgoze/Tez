const uploadZone = document.getElementById("uploadZone");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const btnPickImage = document.getElementById("btnPickImage");
const btnAnalyze = document.getElementById("btnAnalyze");
const resultText = document.getElementById("resultText");
const resultScore = document.getElementById("resultScore");

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    previewImage.src = event.target?.result;
    previewImage.style.display = "block";
    uploadPlaceholder.style.display = "none";
    btnAnalyze.disabled = false;
    resultText.textContent = "Hazır: Görsel yüklendi. Analizi başlatabilirsiniz.";
    resultText.className = "result-text";
    resultScore.textContent = "Güven skoru: —";
  };
  reader.readAsDataURL(file);
}

function processFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    resultText.textContent = "Lütfen geçerli bir görsel dosyası yükleyin.";
    resultText.className = "result-text result-positive";
    return;
  }

  showPreview(file);
}

uploadZone.addEventListener("click", () => imageInput.click());
btnPickImage.addEventListener("click", (event) => {
  event.stopPropagation();
  imageInput.click();
});

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  processFile(file);
});

uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadZone.classList.remove("dragover");
  const [file] = event.dataTransfer.files;
  processFile(file);
});

btnAnalyze.addEventListener("click", () => {
  // TODO: Buraya model çağrısı entegre edilecek.
  // Geçici demo davranışı: örnek bir çıktı göster.
  const confidence = (Math.random() * 0.2 + 0.78).toFixed(2);
  const positive = Math.random() > 0.5;

  resultText.textContent = positive
    ? "Sonuç: Hipertansif retinopati şüphesi mevcut."
    : "Sonuç: Hipertansif retinopati bulgusu saptanmadı.";
  resultText.className = `result-text ${positive ? "result-positive" : "result-negative"}`;
  resultScore.textContent = `Güven skoru: ${confidence}`;
});
