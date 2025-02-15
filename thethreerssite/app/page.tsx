"use client"

import { useState, useRef, useCallback } from "react"
import Webcam from "react-webcam"
import { Button } from "@/components/ui/button"
import { MapIcon, Camera, RefreshCw } from 'lucide-react'

const mirrorClass = "scale-x-[-1]"

export default function TrashClassification() {
  const webcamRef = useRef<Webcam>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [isCapturing, setIsCapturing] = useState(true)

  const capture = useCallback(() => {
    if (isCapturing) {
      const imageSrc = webcamRef.current?.getScreenshot()
      if (imageSrc) {
        // Create a new image to flip horizontally
        const img = new Image()
        img.onload = () => {
          const canvas = document.createElement('canvas')
          canvas.width = img.width
          canvas.height = img.height
          const ctx = canvas.getContext('2d')
          if (ctx) {
            ctx.scale(-1, 1) // Flip horizontally
            ctx.drawImage(img, -img.width, 0)
            setCapturedImage(canvas.toDataURL())
          }
          setIsCapturing(false)
          console.log("Image captured and mirrored. Ready for classification.")
        }
        img.src = imageSrc
      }
    } else {
      setCapturedImage(null)
      setIsCapturing(true)
    }
  }, [isCapturing])

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Trash Classification System</h1>
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1">
          <div className="relative aspect-video">
            {isCapturing ? (
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className={`rounded-lg w-full h-full object-cover ${mirrorClass}`}
                mirrored={true}
              />
            ) : (
              <img
                src={capturedImage || "/placeholder.svg"}
                alt="Captured"
                className={`rounded-lg w-full h-full object-cover ${mirrorClass}`}
              />
            )}
          </div>
          <Button onClick={capture} className="mt-2 w-full">
            {isCapturing ? (
              <>
                <Camera className="mr-2 h-4 w-4" /> Capture Image
              </>
            ) : (
              <>
                <RefreshCw className="mr-2 h-4 w-4" /> Take Another Picture
              </>
            )}
          </Button>
          <p className="mt-2 text-center text-sm text-gray-600">
            {isCapturing
              ? "Live webcam feed. Click the button to capture and classify trash."
              : "Image captured. Click the button to take another picture."}
          </p>
        </div>
        <div className="flex-1">
          <div className="bg-gray-200 rounded-lg aspect-video flex items-center justify-center">
            <MapIcon className="h-16 w-16 text-gray-400" />
          </div>
          <p className="mt-2 text-center text-sm text-gray-600">Map placeholder. Will show nearby recycling centers.</p>
        </div>
      </div>
      {capturedImage && !isCapturing && (
        <div className="mt-4">
          <h2 className="text-xl font-semibold mb-2">Classification Result</h2>
          <p className="text-gray-600">Classification results will appear here after processing the image.</p>
        </div>
      )}
    </div>
  )
}

