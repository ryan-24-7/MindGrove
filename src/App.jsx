import React, { useState } from 'react';
import { Button, Typography, Box, LinearProgress, Container, Paper, Snackbar, CssBaseline, TextField } from '@mui/material';
import { CloudUpload as CloudUploadIcon } from '@mui/icons-material';
import MuiAlert from '@mui/material/Alert';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import './App.css';
import axios from 'axios';

// Create a theme instance with custom colors.
const theme = createTheme({
  palette: {
    primary: {
      main: '#294c60',
    },
    secondary: {
      main: '#5db5a4',
    },
    background: {
      default: '#f0f0f0',
    },
  },
  typography: {
    fontFamily: '"Nunito", sans-serif',
    h4: {
      fontWeight: 600,
    },
    button: {
      textTransform: 'none',
    },
  },
  shape: {
    borderRadius: 8, // Rounded corners for elements
  },
  transitions: {
    duration: {
      enteringScreen: 500, // Smooth transitions for entering screen
      leavingScreen: 300, // Smooth transitions for leaving screen
    },
  },
});


const Alert = React.forwardRef(function Alert(props, ref) {
  return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
});

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('info');
  const [plotUrl, setPlotUrl] = useState(null);
  const [showWelcome, setShowWelcome] = useState(true);
  const [showJournal, setShowJournal] = useState(false);
  const [journalText, setJournalText] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadProgress(0);
    setPlotUrl(null);
  };

  const handleUpload = () => {
    if (!selectedFile) {
      handleSnackbarOpen('Please select a file before uploading.', 'warning');
      return;
    }

    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    axios.post('http://localhost:5000/upload', formData, {
      onUploadProgress: progressEvent => {
        let percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);
      }
    })
      .then(response => {
        // Assuming the response contains { image: 'data:image/png;base64,...', text: 'Recognized text' }
        console.log(response.data.image)
        setPlotUrl(response.data.image);
        setJournalText(response.data.text); // Update journal text state
        handleSnackbarOpen('File uploaded and processed successfully!', 'success');
        setShowJournal(true);
      })
      .catch(error => {
        console.error('Upload error:', error);
        handleSnackbarOpen('An error occurred during the upload.', 'error');
        setUploadProgress(0);
      })
      .finally(() => {
        setIsUploading(false);
      });
  };

  const handleSnackbarOpen = (message, severity) => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setOpenSnackbar(true);
  };

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setOpenSnackbar(false);
  };

  const Alert = React.forwardRef(function Alert(props, ref) {
    return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
  });

  const WelcomeScreen = () => (
    <Box className="App"
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 2,
        animation: 'fadeIn 1.5s ease-in-out',
      }}
    >
      <Typography variant="h1" gutterBottom sx={{ color: 'black', fontStyle: 'italic', fontFamily: 'Playfair Display' }}>Welcome to MindGrove</Typography>
      <Button variant="text" onClick={() => setShowWelcome(false)} sx={{ fontFamily: 'Playfair Display' }}>Click here to Begin</Button>
    </Box>
  );

  const JournalBox = () => (
    <Box sx={{ my: 2, width: '100%' }}>
    <Typography variant="h6" gutterBottom>Journal</Typography>
    <TextField
      multiline
      rows={4}
      variant="outlined"
      fullWidth
      value={journalText}
      disabled={true}
    />
  </Box>
  );


  return (
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* Normalize the styling */}
      {showWelcome ? (
        <WelcomeScreen />
      ) : (
        <Box className="App" sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'flex-start', // Adjust alignment to ensure content starts from the top
          alignItems: 'center',
          p: 2,
          overflow: 'auto', // Ensure content can scroll if it exceeds the viewport height
          maxHeight: '100vh', // Ensure the container does not exceed the viewport height
          width: '100%', // Ensure the container takes up the full width
        }}>
          <Container component="main" maxWidth="sm" sx={{ mb: 4 }}>
            <Paper elevation={6} sx={{
              my: { xs: 2, md: 6 },
              p: { xs: 2, md: 3 },
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              transition: 'transform 0.3s ease-in-out',
              '&:hover': {
                transform: 'scale(1.02)',
                boxShadow: theme.shadows[1],
              }
            }}>
              <Typography variant="h4" color="primary" gutterBottom>
                MindGrove: Speech to Emotion
              </Typography>
              <Box sx={{ width: '100%', my: 2 }}>
                <Button variant="contained" component="label" startIcon={<CloudUploadIcon />} disabled={isUploading} fullWidth>
                  Choose File
                  <input type="file" hidden onChange={handleFileChange} accept=".mp4, .mp3" />
                </Button>
                {selectedFile && <Typography variant="subtitle1" sx={{ mt: 2 }}>{selectedFile.name}</Typography>}
                {isUploading && <LinearProgress variant="determinate" value={uploadProgress} sx={{ width: '100%', my: 2 }} />}
                <Button variant="contained" color="secondary" onClick={handleUpload} disabled={!selectedFile || isUploading} fullWidth sx={{ mt: 2 }}>
                  Upload
                </Button>
              </Box>
              {showJournal && <JournalBox />}
              {plotUrl && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="h6">Analysis Plot:</Typography>
                  <img src={plotUrl} alt="Analysis Plot" style={{ maxWidth: '100%', height: 'auto' }} />
                </Box>
              )}

            </Paper>
          </Container>
          <Snackbar open={openSnackbar} autoHideDuration={6000} onClose={handleSnackbarClose}>
            <Alert onClose={handleSnackbarClose} severity={snackbarSeverity} sx={{ width: '100%' }}>
              {snackbarMessage}
            </Alert>
          </Snackbar>
        </Box>)}
    </ThemeProvider>
  );
}

export default App;
