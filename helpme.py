class helpyou:
	"""
	PreRequisite : Knowlege of Python (Basic[class, modules, function]), Pytorch, Neural Network, Activation Function.
	
	Installation : pytorch(CUDA 10.2), torchvision, pillow, mpi4py, numpy.
	
	To Train and Get Generated Images Run design.py
	

	************************************************************************************************************************************************************************************

	
	PAPER :
		GAN, MD-GAN	:
			Generative Adversarial Network, Multi-Discriminator GenerativeAdversarial Networks for Distributed Datasets


		AUTHORS		:
			PAPER I - [ Ian J. Goodfellow,  Jean Pouget-Abadie∗, Mehdi Mirza, Bing Xu, David Warde-Farley,Sherjil Ozair†, Aaron      Courville, Yoshua Bengio‡ ]
  			PAPER II - [ Corentin Hardy, Erwan Le Merrer, Bruno Sericola ]


		LINK		:
			PAPER I  :  https://arxiv.org/pdf/1406.2661.pdf
        		PAPER II : https://arxiv.org/pdf/1811.03850v2.pdf
	************************************************************************************************************************************************************************************


	Main libraries :

		numpy			:	It's a multidimensional Array.

		mpi4py			:	MPI for Python supports convenient, pickle-based communication of generic Python object as well as fast, near C-speed, direct array data communication of buffer-provider objects.

		torch			:	An open source machine learning framework that accelerates the path from research prototyping to production deployment.(Official site)	

		torch.nn:		:	Base class for all neural network modules, our models is also subclass this class.		

		torch.optim		:	torch.optim is a package implementing various optimization algorithms. Most commonly used methods are already supported, and the interface is general enough,
						 so that more sophisticated ones can be also easily integrated in the future.

		torch.utils.data	:	It represents a python iterable over a dataset.

		torch.nn.parallel	:	This container parallelizes the application of the given module by splitting the input across the specified devices by chunking in the batch dimension.

		torchvision		:	The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.

	Other libraries :

		random, os.


	************************************************************************************************************************************************************************************


	Important Variables :

		size			:	Represt the size of message pass to suffle the Discriminator using peer2peer fashion.

		rank			:	Push the Discriminator to use respective position (bcz we are using two discriminator, rank discriminator by 1/0 (1 : run next, 0: currently running))
		
		datasets		:	This varible is heart of the programme bcz this will download and store the dataset. 

		dataloader		:	At the heart of PyTorch data loading utility is the torch.utils.data.DataLoader class, We are using one HUGE DATASET CIFAR10 in this implimentation.



	************************************************************************************************************************************************************************************


	Main Class :

		G()			:	This class is used to create generator Neural Network by using torch.nn.Module class.

		D()			:	This class is used to create Discriminator Neural Network by using torch.nn.Module class.


	************************************************************************************************************************************************************************************


	Main function :

		copyGenerator		:	Create a copy of generator to get the feedback of the generator to learn from them.

		shuffleDiscriminators	:	Shuffle the discriminator on the basis of rank after every 2 epochs.


	************************************************************************************************************************************************************************************


	Sudo code of implimentation :

		Algorithm  1MD-GAN algorithm
				1:procedureWORKER(C,Bn,I,L,b)
				2:	InitializeθnforDn
				3:	fori←1toIdo
				4:		X(r)n←SAMPLES(Bn,b)
				5:		X(g)n,X(d)n←RECEIVEBATCHES(C)
				6:		forl←0toLdo
				7:			Dn←DISCLEARNINGSTEP(Jdisc,Dn)
				8:		end for
				9:	Fn←{∂ ̃B(X(g)n)∂xi|xi∈X(g)n}
				10:	SEND(C,Fn).SendFnto server
				11:	ifimod (mEb) = 0then
				12:		Dn←SWAP(Dn)
				13:	end if
				14:	end for
				15:end procedure
				16:
				17:procedureSWAP(Dn)
				18:	Wl←GETRANDOMWORKER()
				19:	SEND(Wl,Dn).SendDnto workerWl.
				20:	Dn←RECEIVED().Receive a new discriminatorfrom another worker.
				21:	ReturnDn
				22:end procedure
				23:
				24:procedureSERVER(k,I).Server C
				25:	InitializewforG
				26:	fori←1toIdo
				27:		forj←0tokdo
				28:			Zj←GAUSSIANNOISE(b)
				29:			X(j)←{Gw(z)|z∈Zj}
				30:		end for
				31:		X(d)1,...,X(d)n←SPLIT(X(1),...,X(k))
				32:		X(g)1,...,X(g)n←SPLIT(X(1),...,X(k))
				33:		for n←1toNdo
				34:			SEND(Wn,(X(d)n,X(g)n))
				35:		end for
				36:		F1,...,FN←GETFEEDBACKFROMWORKERS()
				37:		Compute∆waccording toF1,...,FN
				38:		for wi∈w do
				39:			wi←wi+ADAM(∆wi)
				40:		end for
				41:	end for
				42:end procedure

	"""
	
		
print(help(helpyou))
