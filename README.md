
## MOONDREAM 2 


https://huggingface.co/vikhyatk/moondream2 

<img width="1431" height="634" alt="PNG image" src="https://github.com/user-attachments/assets/da4f9c8a-0b33-4115-802b-dab8bde4b588" />



I took a fully different approach and planned according to the huggingface model entirely; and then being able to still use the Docker commands given in the class instructions like using the dockerfile, dockerfile.cpu, manage vector db,  environment file, streamlit interface, and using the docker commands also given in the instructions like docker compose -f docker-compose.cpu.yml up. what differed is that some of my files-- and more specifically my script-- was completely rewritten to directly integrate Moondream2 from Hugging Face instead of using ramalama's API approach-- so it was able to ass vision capabilities to process both images and text documents simultaneously. 


Sadly, my computer did not have the capability to run the model; i have an M2 and especially because i'd only be able to use CPU with docker-- i was able to fully get to local host, upload an image or text file and have it process, but then when i'd ask a question about it, i'd get an error (which through much research) essentially said that my CPU couldn't handle the model. i ended up adding measures to try and reduce some aspects (optimizing float16 vs float32) but it still wouldn't work for images. i tried it with a text file and asked about it has taken... probably... at least 30 minutes to run... and as of now still has not given me an answer. 


thank you so much for a great semester!!!
