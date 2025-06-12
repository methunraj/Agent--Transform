import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { FileUploader } from "@/components/files/FileUploader";
import { DirectorySelector } from "@/components/files/DirectorySelector";
import { SelectedFilesList } from "@/components/files/SelectedFilesList";

export default function FileManagementPage() {
  return (
    <div className="flex-1 space-y-6">
      <div className="flex items-center justify-between space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">File Management</h1>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Upload Files</CardTitle>
            <CardDescription>
              Upload individual documents for data extraction. Supported types: TXT, PDF, DOCX, JPG, PNG.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FileUploader />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Select Directory</CardTitle>
            <CardDescription>
              Specify a local directory to process all supported files within it.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <DirectorySelector />
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Selected Files for Processing</CardTitle>
          <CardDescription>
            Review the list of files currently selected for the extraction job.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <SelectedFilesList />
        </CardContent>
      </Card>
    </div>
  );
}
