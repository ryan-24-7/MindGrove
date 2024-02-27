import React, {useState} from "react";
import { Button } from "react-bootstrap";

function FileUpload({ onFileSelect }) {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
    };

    const onSubmit = () => {
        if (selectedFile) {
            onFileSelect(selectedFile);
        } else {
            alert('Please select a file before submitting.');
        }
    };

    return (
        <div className="file-uploader">
            <input type="file" accept=".mp3" onChange={handleFileChange} className="file-input" />
            <Button type="submit" onClick={onSubmit}className='submit-button'>
                Analyze File
            </Button>
        </div>
    );
}

export default FileUpload;