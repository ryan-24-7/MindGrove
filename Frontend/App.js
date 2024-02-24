import React, { useState } from 'react';
import { Button, Typography, Box, LinearProgress, Container, Paper, Snackbar, CssBaseline } from '@mui/material';
import { CloudUpload as CloudUploadIcon } from '@mui/icons-material';
import MuiAlert from '@mui/material/Alert';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import './App.css';

// Create a theme instance with custom colors.
const theme = createTheme({
  palette: {
    primary: {
      main: '#294c60', // Adjusted primary color for better harmony
    },
    secondary: {
      main: '#5db5a4', // Complementary secondary color
    },
    background: {
      default: '#e0f7fa',
    },
  },
  typography: {
    fontFamily: '"Montserrat", sans-serif', // A fancy font from Google Fonts
    h4: {
      fontWeight: 600,
    },
    button: {
      textTransform: 'none',
    },
  },
});

// Custom Alert component for Snackbar
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

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadProgress(0);
  };

  const handleUpload = () => {
    if (!selectedFile) {
      handleSnackbarOpen('Please select a file before uploading.', 'warning');
      return;
    }

    const formData = new FormData();
    formData.append("speechFile", selectedFile);

    setIsUploading(true);
    
    fetch('/api/upload', {
      method: 'POST',
      body: formData,
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      handleSnackbarOpen("File uploaded successfully!", "success");
      setUploadProgress(100);
    })
    .catch(error => {
      handleSnackbarOpen("An error occurred during the upload.", "error");
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

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* Normalize the styling */}
      <Box className="App" sx={{ height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', p: 2 }}>
        <Container component="main" maxWidth="sm">
          <Paper elevation={6} sx={{ py: 5, px: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', backgroundColor: 'rgba(255, 255, 255, 0.8)' }}>
            <Typography variant="h4" color="primary" gutterBottom>
              Speech Recognition Upload
            </Typography>
            <Box sx={{ width: '100%', my: 2 }}>
              <Button variant="contained" component="label" startIcon={<CloudUploadIcon />} disabled={isUploading} fullWidth>
                Choose File
                <input type="file" hidden onChange={handleFileChange} accept=".mp3"/>
              </Button>
              {selectedFile && <Typography variant="subtitle1" sx={{ mt: 2 }}>{selectedFile.name}</Typography>}
              {isUploading && <LinearProgress variant="determinate" value={uploadProgress} sx={{ width: '100%', my: 2 }}/>}
              <Button variant="contained" color="secondary" onClick={handleUpload} disabled={!selectedFile || isUploading} fullWidth sx={{ mt: 2 }}>
                Upload
              </Button>
            </Box>
          </Paper>
        </Container>
        <Snackbar open={openSnackbar} autoHideDuration={6000} onClose={handleSnackbarClose}>
          <Alert onClose={handleSnackbarClose} severity={snackbarSeverity} sx={{ width: '100%' }}>
            {snackbarMessage}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;