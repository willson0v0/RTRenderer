#include "RTRendererCUDAQT.h"
#include <QImage>
#include <QPainter>
#include "ui_RTRendererCUDAQT.h"
#include <QtWidgets/QApplication>
#include "consts.h"
#include <string>
#include <sstream>
#include <algorithm>



//认真布局了
//改名字 一一对应 Lab->Lab StartButton->StartButton StopButton ParameterText->ParameterText  Para_Stop->Para_Stop
//函数名

RTRendererCUDAQT::RTRendererCUDAQT(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setGeometry(10, 10, MAX_X + 400, MAX_Y + 400);

	Lab = new QLabel(this);
	Lab->setGeometry(20, 20, MAX_X, MAX_Y);

	logText = new QTextEdit(this);
	logText->setGeometry(MAX_X + 50, 20, 300, MAX_Y);

	StartButton = new QPushButton("Start", this);
	StartButton->setGeometry(MAX_X + 50,MAX_Y + 40 ,100 ,20 );

	StopButton = new QPushButton("Stop", this);
	StopButton->setGeometry(MAX_X + 50, MAX_Y + 80 , 100, 20);

	Updater = new QPushButton("Update", this);
	Updater->setGeometry(MAX_X + 50, MAX_Y + 120, 100, 20);

	Discarder = new QPushButton("Discard", this);
	Discarder->setGeometry(MAX_X + 50, MAX_Y + 160, 100, 20);


	
	paraTargetSPP = new QLineEdit(this);
	paraClipUpperbound = new QLineEdit(this);
	paraLookatX = new QLineEdit(this);
	paraLookatY = new QLineEdit(this);
	paraLookatZ = new QLineEdit(this);

	paraLookfromX = new QLineEdit(this);
	paraLookfromY = new QLineEdit(this);
	paraLookfromZ = new QLineEdit(this);
	paraVupX = new QLineEdit(this);
	paraVupY = new QLineEdit(this);
	paraVupZ = new QLineEdit(this);
	paraFocusDist = new QLineEdit(this);
	paraAperture = new QLineEdit(this);
	paraFov = new QLineEdit(this);



	paraClipUpperbound->setGeometry(120, MAX_Y + 20, 80, 20);
	paraTargetSPP->setGeometry(20, MAX_Y + 20, 80, 20);
	paraLookatX->setGeometry(20, MAX_Y + 60, 80, 20);
	paraLookatY->setGeometry(120, MAX_Y + 60, 80, 20);
	paraLookatZ->setGeometry(220, MAX_Y + 60, 80, 20);

	paraLookfromX->setGeometry(20, MAX_Y + 100, 80, 20);
	paraLookfromY->setGeometry(120, MAX_Y + 100, 80, 20);
	paraLookfromZ->setGeometry(220, MAX_Y + 100, 80, 20);
	paraVupX->setGeometry(20, MAX_Y + 140, 80, 20);
	paraVupY->setGeometry(120, MAX_Y + 140, 80, 20);
	paraVupZ->setGeometry(220, MAX_Y + 140, 80, 20);
	paraFocusDist->setGeometry(20, MAX_Y + 180, 80, 20);
	paraAperture->setGeometry(120, MAX_Y + 180, 80, 20);
	paraFov->setGeometry(220, MAX_Y + 180, 80, 20);
	


	labTargetSPP = new QLabel(this);
	labClipUpperbound = new QLabel(this);
	labLookatX = new QLabel(this);
	labLookatY = new QLabel(this);
	labLookatZ = new QLabel(this);

	labLookfromX = new QLabel(this);
	labLookfromY = new QLabel(this);
	labLookfromZ = new QLabel(this);
	labVupX = new QLabel(this);
	labVupY = new QLabel(this);
	labVupZ = new QLabel(this);
	labFocusDist = new QLabel(this);
	labAperture = new QLabel(this);
	labFov = new QLabel(this);


	labTargetSPP->setGeometry(20, MAX_Y + 40, 80, 20);
	labClipUpperbound->setGeometry(120, MAX_Y + 40, 80, 20);
	labLookatX->setGeometry(20, MAX_Y + 80, 80, 20);
	labLookatY->setGeometry(120, MAX_Y + 80, 80, 20);
	labLookatZ->setGeometry(220, MAX_Y + 80, 80, 20);


	labLookfromX->setGeometry(20, MAX_Y + 120, 80, 20);
	labLookfromY->setGeometry(120, MAX_Y + 120, 80, 20);
	labLookfromZ->setGeometry(220, MAX_Y + 120, 80, 20);
	labVupX->setGeometry(20, MAX_Y + 160, 80, 20);
	labVupY->setGeometry(120, MAX_Y + 160, 80, 20);
	labVupZ->setGeometry(220, MAX_Y + 160, 80, 20);
	labFocusDist->setGeometry(20, MAX_Y + 200, 80, 20);
	labAperture->setGeometry(120, MAX_Y + 200, 80, 20);
	labFov->setGeometry(220, MAX_Y + 200, 80, 20);


	labTargetSPP->setText("TargetSPP");
	labClipUpperbound->setText("ClipUpperbound");
	labLookatX->setText("LookatX");
	labLookatY->setText("LookatY");
	labLookatZ->setText("LookatZ");

	labLookfromX->setText("LookfromX");
	labLookfromY->setText("LookfromY");
	labLookfromZ->setText("LookfromZ");
	labVupX->setText("VupX");
	labVupY->setText("VupY");
	labVupZ->setText("VupZ");
	labFocusDist->setText("FocusDist");
	labAperture->setText("Aperture");
	labFov->setText("Fov");



	looper = new LoopThread(this);
	connect(StartButton, SIGNAL(clicked()), this, SLOT(Startear()));
	connect(looper, SIGNAL(refresh_flag()), this, SLOT(refresh()));
	connect(StopButton, SIGNAL(clicked()), this, SLOT(Stop()));
	connect(Updater, SIGNAL(clicked()), this, SLOT(setParameter()));
	connect(Discarder, SIGNAL(clicked()), this, SLOT(discardParameter()));
}



void LoopThread::run()
{
	kernel();
}


//给自己加个模板，还有其他一些东西。
void LoopThread::PrintMessege()
{
	emit info_flag();
}

void LoopThread::checkBreak()
{
	if (this->frameCount * SPP >= this->targetSPP)
	{
		this->break_flag = 1;
		
	}
}

void RTRendererCUDAQT::setLabel(int index, int x, int y, std::string name)
{
	labParameter[index] = new QLabel(this);
	lineParameter[index] = new  QLineEdit(this);
	labParameter[index]->setGeometry(x, y, 80, 20);

}


void RTRendererCUDAQT::initialization()
{
	this->looper->break_flag = 0;
	this->looper->end_flag = 0;
	this->looper->targetClipUpperbound = 1.0;
	this->looper->targetSPP = INT_MAX;//默认是最大值，不依靠它提供break

	this->looper->Lookat.e[0] = 0;
	this->looper->Lookat.e[1] = 0;
	this->looper->Lookat.e[2] = 0;

	this->looper->Lookfrom.e[0] = 13.0;
	this->looper->Lookfrom.e[1] = 2.0;
	this->looper->Lookfrom.e[2] = 3.0;

	this->looper->Vup.e[0] = 0;
	this->looper->Vup.e[1] = 1.0;
	this->looper->Vup.e[2] = 0;

	this->looper->FocusDist = 9.5;
	this->looper->Aperture = 0.1;
	this->looper->Fov = 30.0;


	this->discardParameter();
}

void RTRendererCUDAQT::Startear()
{
	this->initialization();
	this->ShowPara();
	looper->start();
}

void RTRendererCUDAQT::Stop()
{
	this->looper->break_flag = 1;
	this->looper->end_flag = 1;
}

void RTRendererCUDAQT::setParameter()
{
	this->looper->targetSPP = this->paraTargetSPP->text().toInt();
	this->looper->targetClipUpperbound = this->paraClipUpperbound->text().toFloat();

	this->looper->Lookat.e[0] = this->paraLookatX->text().toFloat();
	this->looper->Lookat.e[1] = this->paraLookatY->text().toFloat();
	this->looper->Lookat.e[2] = this->paraLookatZ->text().toFloat();

	this->looper->Lookfrom.e[0] = this->paraLookfromX->text().toFloat();
	this->looper->Lookfrom.e[1] = this->paraLookfromY->text().toFloat();
	this->looper->Lookfrom.e[2] = this->paraLookfromZ->text().toFloat();

	this->looper->Vup.e[0] = this->paraVupX->text().toFloat();
	this->looper->Vup.e[1] = this->paraVupY->text().toFloat();
	this->looper->Vup.e[2] = this->paraVupZ->text().toFloat();

	this->looper->FocusDist = this->paraFocusDist->text().toInt();
	this->looper->Aperture = this->paraAperture->text().toInt();
	this->looper->Fov = this->paraFov->text().toInt();

	this->ShowPara();
	this->looper->frameCount = 0;
	this->looper->break_flag = 1;

}

void RTRendererCUDAQT::ShowPara()
{
	std::string str = "SPP Threshold =" + std::to_string(this->looper->targetSPP);
	this->logText->append(QString::fromStdString(str));
	str = "Clip Upperbound =" + std::to_string(this->looper->targetClipUpperbound);
	this->logText->append(QString::fromStdString(str));
	str = "LookatX =" + std::to_string(this->looper->Lookat.e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "LookatY =" + std::to_string(this->looper->Lookat.e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "LookatZ =" + std::to_string(this->looper->Lookat.e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromX =" + std::to_string(this->looper->Lookfrom.e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromY =" + std::to_string(this->looper->Lookfrom.e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromZ =" + std::to_string(this->looper->Lookfrom.e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "VupX =" + std::to_string(this->looper->Vup.e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "VupY =" + std::to_string(this->looper->Vup.e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "VupZ =" + std::to_string(this->looper->Vup.e[2]);
	this->logText->append(QString::fromStdString(str));

	str = "FocusDist =" + std::to_string(this->looper->FocusDist);
	this->logText->append(QString::fromStdString(str));
	str = "Aperture =" + std::to_string(this->looper->Aperture);
	this->logText->append(QString::fromStdString(str));
	str = "Fov =" + std::to_string(this->looper->Fov);
	this->logText->append(QString::fromStdString(str));

}

void RTRendererCUDAQT::discardParameter()
{
	this->paraTargetSPP->setText(QString::fromStdString(std::to_string(this->looper->targetSPP)));
	this->paraClipUpperbound->setText(QString::fromStdString(std::to_string(this->looper->targetClipUpperbound)));
	this->paraLookatX->setText(QString::fromStdString(std::to_string(this->looper->Lookat.e[0])));
	this->paraLookatY->setText(QString::fromStdString(std::to_string(this->looper->Lookat.e[1])));
	this->paraLookatZ->setText(QString::fromStdString(std::to_string(this->looper->Lookat.e[2])));
	this->paraLookfromX->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom.e[0])));
	this->paraLookfromY->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom.e[1])));
	this->paraLookfromZ->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom.e[2])));
	this->paraVupX->setText(QString::fromStdString(std::to_string(this->looper->Vup.e[0])));
	this->paraVupY->setText(QString::fromStdString(std::to_string(this->looper->Vup.e[1])));
	this->paraVupZ->setText(QString::fromStdString(std::to_string(this->looper->Vup.e[2])));
	this->paraFocusDist->setText(QString::fromStdString(std::to_string(this->looper->FocusDist)));
	this->paraAperture->setText(QString::fromStdString(std::to_string(this->looper->Aperture)));
	this->paraFov->setText(QString::fromStdString(std::to_string(this->looper->Fov)));
}
