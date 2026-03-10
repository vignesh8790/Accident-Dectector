const path = require('path');
const fs = require('fs');
const multer = require('multer');
const Analysis = require('../models/Analysis');

const SAMPLE_VIDEOS_DIR = path.join(__dirname, '..', '..', 'sample-videos');
const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');

// Ensure uploads directory exists
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const allowed = ['.mp4', '.avi', '.mov', '.mkv'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) cb(null, true);
    else cb(new Error('Only video files (mp4, avi, mov, mkv) are allowed.'));
  },
  limits: { fileSize: 500 * 1024 * 1024 } // 500MB
});

exports.uploadMiddleware = upload.single('video');

exports.streamVideo = (req, res) => {
  try {
    const { filename } = req.params;
    const safeName = path.basename(filename);
    let filePath = path.join(SAMPLE_VIDEOS_DIR, safeName);

    // Also check uploads directory
    if (!fs.existsSync(filePath)) {
      filePath = path.join(UPLOADS_DIR, safeName);
    }

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'Video file not found.' });
    }

    const extName = path.extname(filePath).toLowerCase();
    const mimeTypes = {
      '.mp4': 'video/mp4',
      '.webm': 'video/webm',
      '.avi': 'video/x-msvideo',
      '.mov': 'video/quicktime',
      '.mkv': 'video/x-matroska'
    };
    const contentType = mimeTypes[extName] || 'video/mp4';

    const stat = fs.statSync(filePath);
    const fileSize = stat.size;
    const range = req.headers.range;

    if (range) {
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = end - start + 1;

      const file = fs.createReadStream(filePath, { start, end });
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunkSize,
        'Content-Type': contentType,
      });
      file.pipe(res);
    } else {
      res.writeHead(200, {
        'Content-Length': fileSize,
        'Content-Type': contentType,
      });
      fs.createReadStream(filePath).pipe(res);
    }
  } catch (error) {
    res.status(500).json({ error: 'Failed to stream video.' });
  }
};

exports.uploadVideo = async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No video file uploaded.' });
    
    const { spawn } = require('child_process');
    const inputPath = req.file.path;
    const safeName = req.file.filename;
    // We transcode to an MP4 with H.264 (avc1) for maximum browser compatibility
    const previewFilename = `preview-${path.parse(safeName).name}.mp4`;
    const outputPath = path.join(UPLOADS_DIR, previewFilename);
    const pythonScript = path.join(__dirname, '..', '..', '..', 'Engine', 'codes', 'preprocess.py');
    const cwd = path.dirname(pythonScript);

    console.log(`[Transcode] Starting for ${safeName}...`);
    const pythonProcess = spawn('python', [pythonScript, inputPath, outputPath], { cwd });

    pythonProcess.on('error', (err) => {
      console.error(`[Transcode] Spawn error for ${safeName}:`, err);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Failed to start video processing engine.' });
      }
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        console.log(`[Transcode] Finished successfully for ${safeName}`);
        res.json({
          message: 'Video uploaded and prepared successfully.',
          filename: req.file.filename,
          previewFilename: previewFilename,
          path: `/api/videos/stream/${previewFilename}`
        });
      } else {
        console.error(`[Transcode] Failed for ${safeName} with code ${code}`);
        // Fallback to original if transcoding fails (even if it might not play)
        res.json({
          message: 'Video uploaded (preparation failed).',
          filename: req.file.filename,
          path: `/api/videos/stream/${req.file.filename}`
        });
      }
    });

  } catch (error) {
    console.error('Upload/Transcode error:', error);
    res.status(500).json({ error: 'Failed to upload or prepare video.' });
  }
};

exports.analyzeVideo = (req, res) => {
  const { filename } = req.body;
  if (!filename) return res.status(400).json({ error: 'Filename is required for analysis.' });

  const safeName = path.basename(filename);
  const filePath = path.join(UPLOADS_DIR, safeName);

  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'Video file not found.' });
  }

  // Path to the main Engine inference script
  const pythonScript = path.join(__dirname, '..', '..', '..', 'Engine', 'codes', 'test_inference_lite.py');
  
  // We need to run it in the Engine/codes directory so it finds models/yolo weights correctly
  const cwd = path.dirname(pythonScript);
  
  const { spawn } = require('child_process');
  
  // We're using the virtual env python if it exists, otherwise just global python
  const pythonPath = 'python'; 
  
  // Create an output path for the annotated video
  const outFileName = `annotated-${safeName}`;
  const outFilePath = path.join(UPLOADS_DIR, outFileName);

  const pythonProcess = spawn(pythonPath, [pythonScript, '--video', filePath, '--output', outFilePath], { cwd });

  pythonProcess.on('error', (err) => {
    console.error(`[AI Error] Spawn error:`, err);
    if (!res.headersSent) {
      res.status(500).json({ error: 'AI processing engine is not available or failed to start.' });
    }
  });

  const markers = [];
  let stdoutBuffer = '';

  pythonProcess.stdout.on('data', (data) => {
    stdoutBuffer += data.toString();
    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop(); // Keep the incomplete line for the next chunk

    for (const line of lines) {
      if (line.startsWith('FRAME:')) {
        const b64 = line.substring(6).trim();
        // Emit live frame to any connected Socket.IO clients listening to 'video_frame'
        req.app.get('io').emit('video_frame', `data:image/jpeg;base64,${b64}`);
      } else if (line.includes('MARKER:')) {
        const parts = line.split('MARKER:')[1].trim().split(':');
        if (parts.length >= 2) {
          markers.push({
            time: Math.round(parseFloat(parts[0])),
            confidence: Math.round(parseFloat(parts[1])),
            objects: parts.length > 2 ? parts[2].split(',') : ['vehicle']
          });
        }
      } else if (line.trim().length > 0) {
        console.log(`[AI] ${line.trim()}`);
      }
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[AI Error] ${data.toString().trim()}`);
  });

  pythonProcess.on('close', async (code) => {
    console.log(`[AI Analysis] Finished with code ${code}`);
    
    const analysisData = {
      userId: req.user._id,
      fileName: safeName,
      fileSize: fs.statSync(filePath).size,
      duration: 0, // In real app, we would get this from probe or cap
      markers: markers,
      originalVideoPath: `/api/videos/stream/${safeName}`,
      annotatedVideoUrl: `/api/videos/stream/${outFileName}`,
      processedAt: new Date()
    };

    try {
      const savedAnalysis = await Analysis.create(analysisData);
      res.json({
        success: true,
        markers: markers,
        annotatedVideoUrl: `/api/videos/stream/${outFileName}`,
        analysisId: savedAnalysis._id
      });
    } catch (saveError) {
      console.error('Failed to save analysis results:', saveError);
      res.json({
        success: true,
        markers: markers,
        annotatedVideoUrl: `/api/videos/stream/${outFileName}`
      });
    }
  });
};

exports.getAnalysisHistory = async (req, res) => {
  try {
    const history = await Analysis.find({ userId: req.user._id }).sort({ processedAt: -1 });
    res.json({ history });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch analysis history.' });
  }
};

exports.deleteAnalysis = async (req, res) => {
  try {
    const analysis = await Analysis.findOneAndDelete({ _id: req.params.id, userId: req.user._id });
    if (!analysis) return res.status(404).json({ error: 'Analysis not found.' });
    res.json({ message: 'Analysis history entry deleted.' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to delete analysis history.' });
  }
};

exports.listVideos = (req, res) => {
  try {
    const videos = [];
    if (fs.existsSync(SAMPLE_VIDEOS_DIR)) {
      const files = fs.readdirSync(SAMPLE_VIDEOS_DIR).filter(f => /\.(mp4|avi|mov|mkv)$/i.test(f));
      files.forEach(f => videos.push({ name: f, type: 'sample', path: `/api/videos/stream/${f}` }));
    }
    if (fs.existsSync(UPLOADS_DIR)) {
      const files = fs.readdirSync(UPLOADS_DIR).filter(f => /\.(mp4|avi|mov|mkv)$/i.test(f));
      files.forEach(f => videos.push({ name: f, type: 'uploaded', path: `/api/videos/stream/${f}` }));
    }
    res.json({ videos });
  } catch (error) {
    res.status(500).json({ error: 'Failed to list videos.' });
  }
};
