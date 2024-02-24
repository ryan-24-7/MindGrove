const express = require('express');
const multer = require('multer');
const app = express();

// Configure multer for file upload
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/') // Make sure this uploads directory exists
  },
  filename: function (req, file, cb) {
    cb(null, file.fieldname + '-' + Date.now() + '.mp3')
  }
});

const upload = multer({ storage: storage });

// Enable CORS for your frontend domain
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*'); // for simplicity, allow all domains
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

// POST endpoint for file upload
app.post('/api/upload', upload.single('speechFile'), (req, res) => {
  if (req.file) {
    console.log('Received file: ', req.file);
    // Process the file here (if needed)
    res.status(200).json({ message: 'File uploaded successfully', fileName: req.file.filename });
  } else {
    res.status(400).json({ message: 'No file received' });
  }
});

// Start the server
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});