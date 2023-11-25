import { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import { FileInput, Label } from 'flowbite-react';

function App() {

  const [check, setCheck] = useState(false)
  const [picture, setPicture] = useState([])
  const [image1, setImage1] = useState()
  const [image2, setImage2] = useState()
  const [dividerPosition, setDividerPosition] = useState(50);

  useEffect(() => {
    const handleMouseMove = (e) => {
      const container = document.getElementById('image-container');
      if (!container) return;

      const containerRect = container.getBoundingClientRect();
      const newPosition = ((e.clientX - containerRect.left) / containerRect.width) * 100;

      const clampedPosition = Math.max(0, Math.min(100, newPosition));
      setDividerPosition(clampedPosition);
    };

    document.addEventListener('mousemove', handleMouseMove);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  const URL= 'http://127.0.0.1:5000'

  const base64ImagesArray = [];

const handlePost = (event) => {
  const files = event.target.files;

  for (const file of files) {
    const reader = new FileReader();

    reader.onload = async (e) => {
      const base64Image = e.target.result.split(',')[1];
      const imageName = file.name;
      base64ImagesArray.push({ name: imageName, photo: base64Image });
      setImage1(base64ImagesArray.photo)
      setImage2(base64ImagesArray.photo)
      if (base64ImagesArray.length === files.length) {
        // Выполните POST-запрос на сервер, отправив массив на сервер
        axios.post(URL + '/api/photo',  base64ImagesArray)
          .then(response => {
            const data = response.data;
            // setPicture(data)
            setImage1(data.orig)
            setImage2(data.mask)
            console.log(data)
            
          })
          .catch(error => {
            console.error('Error:', error);
          });
      }
    };

    reader.readAsDataURL(file);
  }
};

  return (
    <>
    <div className='bg-black opacity-60 absolute h-[60px] w-[60px] min-w-min z-20 rounded-br-3xl'>
    <Label
        htmlFor="dropzone-file"
        className="dark:hover:bg-bray-800 flex h-full w-full cursor-pointer flex-col items-center justify-center "
      >
        <div className="flex flex-col items-center justify-center">
          <svg
            className=" h-8 w-8 text-gray-500 dark:text-gray-400"
            aria-hidden="true"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 20 16"
          >
            <path
              stroke="currentColor"
              strokeLinecap="round"
              strokeLineJoin="round"
              strokeWidth="2"
              d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
            />
          </svg>
        </div>
        <FileInput id="dropzone-file" className="hidden" onChange={handlePost}/>
      </Label>
    </div>
    <div className="slider-container " id="image-container">
      
      <div className="slider-image">
        <div className="image-wrapper" style={{ clipPath: `inset(0% ${100 - dividerPosition}% 0% 0%)` }}>
          {image1?
            <img
              // src="/orig.jpeg"
              src={'data:image/jpeg;base64,'+image1}
              // alt="Image 1"
              className='w-screen'
            />
            :
            <img
              src="/orig.jpeg"
              // src={'data:image/jpeg;base64,'+image1.photo}
              // alt="Image 1"
              className='w-screen'
            />
          }
          
        </div>
      </div>
      <div
        className="slider-image "
        style={{ position: 'absolute', top: 0, right: 0, zIndex: 10 }}
      >
        <div
          className="image-wrapper"
          style={{
            clipPath: `inset(0% 0% 0% ${dividerPosition}%)`,
            zIndex: 0,
          }}
        >
          {image2?
            <img
              // src="/mask.jpeg"
              src={'data:image/jpeg;base64,'+image2}
              // alt="Image 2"
              className="saturate-200 contrast-125 w-screen"
            />
            :
            <img
              src="/mask.jpeg"
              // src={'data:image/jpeg;base64,'+image2.photo}
              // alt="Image 2"
              className="saturate-200 contrast-125 w-screen"
            />
          }
          
        </div>
      </div>
    </div>
    
    </>
    
  );
}

export default App;
